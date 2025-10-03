import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import random

from models.model import load_pretrained_faster_rcnn


# Category names for activity diagrams
ACTIVITY_DIAGRAM_CLASSES = [
    'activity-diagrams', 'action', 'activity', 'commeent', 'control_flow',
    'control_flowcontrol_flow', 'decision_node', 'exit_node', 'final_flow_node',
    'final_node', 'fork', 'merge', 'merge_noode', 'none', 'object',
    'object_flow', 'signal_recept', 'signal_send', 'start_node', 'text'
]


class ActivityDiagramDataset(Dataset):
    def __init__(self, hf_dataset, transforms=None):
        self.dataset = hf_dataset
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load sample
        sample = self.dataset[idx]
        
        # Get image
        image = sample['image']
        
        # Convert PIL image to tensor
        image = T.ToTensor()(image)
        
        # Get annotations
        objects = sample['objects']
        boxes = torch.as_tensor(objects['bbox'], dtype=torch.float32)
        
        # Convert from [x, y, width, height] to [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + height
        
        # Get labels (add 1 to account for background class at index 0)
        labels = torch.as_tensor(objects['category'], dtype=torch.int64) + 1
        
        # Handle None/null labels (set to background class)
        labels = torch.where(labels == 0, torch.tensor(0), labels)
        
        # Get areas
        areas = torch.as_tensor(objects['area'], dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([sample['image_id']]),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms if any
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


class AugmentationTransform:
    
    def __init__(self, train=True):
        self.train = train
    
    def __call__(self, image, target):
        if not self.train:
            return image, target
        
        # Random horizontal flip
        if random.random() < 0.5:
            image = T.functional.hflip(image)
            boxes = target['boxes']
            # Flip boxes: x_new = width - x_old
            width = image.shape[2]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target['boxes'] = boxes
        
        # Random color jitter (safe for diagrams)
        if random.random() < 0.5:
            color_jitter = T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
            image = color_jitter(image)
        
        # Random rotation (small angles for diagrams)
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            image = T.functional.rotate(image, angle)
            # Note: For simplicity, we skip bbox rotation here
            # In production, you'd rotate boxes too or use albumentation
        
        return image, target


def create_data_loaders(batch_size=4, num_workers=4, use_augmentation=True):
    print("Loading activity-diagrams-qdobr dataset...")
    dataset = load_dataset("Francesco/activity-diagrams-qdobr")
    
    # Create transforms
    train_transform = AugmentationTransform(train=True) if use_augmentation else None
    val_transform = AugmentationTransform(train=False)  # No augmentation for validation
    
    # Create dataset objects
    train_dataset = ActivityDiagramDataset(dataset['train'], transforms=train_transform)
    val_dataset = ActivityDiagramDataset(dataset['validation'], transforms=val_transform)
    test_dataset = ActivityDiagramDataset(dataset['test'], transforms=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for invalid losses
        if not torch.isfinite(losses):
            print(f"Warning: Loss is {losses}, skipping batch")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update metrics
        loss_value = losses.item()
        total_loss += loss_value
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches if num_batches > 0 else 0


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.train()  # Keep in training mode to get losses
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc="Validation")
    
    for images, targets in pbar:
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        with torch.enable_grad():  # Enable grad to compute loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        if torch.isfinite(losses):
            total_loss += losses.item()
            num_batches += 1
        
        pbar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_faster_rcnn(
    num_epochs=50,
    batch_size=4,
    learning_rate=0.005,
    momentum=0.9,
    weight_decay=0.0005,
    lr_step_size=10,
    lr_gamma=0.1,
    num_workers=4,
    checkpoint_dir='checkpoints',
    device=None,
    early_stopping_patience=10,
    use_augmentation=True,
    reduce_lr_on_plateau=True
):
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Faster R-CNN on Activity Diagrams")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
    print(f"Early Stopping Patience: {early_stopping_patience if early_stopping_patience > 0 else 'Disabled'}")
    print(f"LR Reduction on Plateau: {'Enabled' if reduce_lr_on_plateau else 'Disabled'}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load dataset
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        use_augmentation=use_augmentation
    )
    
    # Create model
    # +1 for background class
    num_classes = len(ACTIVITY_DIAGRAM_CLASSES) + 1
    print(f"\nInitializing Faster R-CNN with {num_classes} classes...")
    model = load_pretrained_faster_rcnn(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    if reduce_lr_on_plateau:
        # ReduceLROnPlateau: reduce LR when val loss plateaus
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        print("Using ReduceLROnPlateau scheduler (factor=0.5, patience=5)")
    else:
        # StepLR: reduce LR at fixed intervals
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_step_size,
            gamma=lr_gamma
        )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        
        # Update learning rate
        if reduce_lr_on_plateau:
            lr_scheduler.step(val_loss)  # ReduceLROnPlateau needs the metric
        else:
            lr_scheduler.step()  # StepLR doesn't need metric
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time:.1f}s")
        
        # Save best model and track early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"   Early stopping triggered after {epoch} epochs!")
                print(f"   No improvement for {early_stopping_patience} consecutive epochs.")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                print(f"{'='*60}")
                break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path.name}")
        
        print(f"{'-'*60}\n")
    
    # Save final model
    final_checkpoint_path = checkpoint_dir / 'final_model.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history,
    }, final_checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_checkpoint_path}")
    print(f"{'='*60}\n")
    
    return model, training_history


def load_trained_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    num_classes = len(ACTIVITY_DIAGRAM_CLASSES) + 1
    model = load_pretrained_faster_rcnn(num_classes=num_classes, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on Activity Diagrams')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience (0 to disable)')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--no-reduce-lr', action='store_true', help='Disable LR reduction on plateau')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_faster_rcnn(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping,
        use_augmentation=not args.no_augmentation,
        reduce_lr_on_plateau=not args.no_reduce_lr
    )
    
    print("\nTraining completed successfully!")

