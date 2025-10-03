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

from models.model import load_pretrained_faster_rcnn


# Category names for activity diagrams
ACTIVITY_DIAGRAM_CLASSES = [
    'activity-diagrams', 'action', 'activity', 'commeent', 'control_flow',
    'control_flowcontrol_flow', 'decision_node', 'exit_node', 'final_flow_node',
    'final_node', 'fork', 'merge', 'merge_noode', 'none', 'object',
    'object_flow', 'signal_recept', 'signal_send', 'start_node', 'text'
]


class ActivityDiagramDataset(Dataset):
    """
    PyTorch Dataset wrapper for activity-diagrams-qdobr dataset.
    """
    
    def __init__(self, hf_dataset, transforms=None):
        """
        Args:
            hf_dataset: HuggingFace dataset split (train/validation/test)
            transforms: Optional transforms to apply to images
        """
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
    """
    Custom collate function for DataLoader.
    Since each image has different number of objects, we need custom collation.
    """
    return tuple(zip(*batch))


def create_data_loaders(batch_size=4, num_workers=4):
    """
    Create train and validation data loaders.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading activity-diagrams-qdobr dataset...")
    dataset = load_dataset("Francesco/activity-diagrams-qdobr")
    
    # Create dataset objects
    train_dataset = ActivityDiagramDataset(dataset['train'])
    val_dataset = ActivityDiagramDataset(dataset['validation'])
    test_dataset = ActivityDiagramDataset(dataset['test'])
    
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
    """
    Train for one epoch.
    
    Args:
        model: Faster R-CNN model
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number
        print_freq: Print frequency
    
    Returns:
        Average loss for the epoch
    """
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
    """
    Evaluate the model on validation set.
    
    Args:
        model: Faster R-CNN model
        data_loader: Validation data loader
        device: Device to evaluate on
    
    Returns:
        Average loss
    """
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
    device=None
):
    """
    Complete training pipeline for Faster R-CNN on activity diagrams.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        momentum: SGD momentum
        weight_decay: Weight decay for regularization
        lr_step_size: Step size for learning rate scheduler
        lr_gamma: Multiplicative factor for learning rate decay
        num_workers: Number of data loading workers
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on (auto-detect if None)
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Faster R-CNN on Activity Diagrams")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load dataset
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=batch_size,
        num_workers=num_workers
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
        lr_scheduler.step()
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
        
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
            print(f"  💾 Saved checkpoint: {checkpoint_path.name}")
        
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
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
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
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_faster_rcnn(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print("\n🎉 Training completed successfully!")

