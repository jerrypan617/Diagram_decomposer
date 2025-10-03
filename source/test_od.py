import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
from collections import defaultdict
import cv2

# OCR imports
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("Warning: PaddleOCR not available. OCR functionality will be disabled.")
    print("Install with: pip install paddlepaddle paddleocr")

from train_od import (
    load_trained_model, 
    create_data_loaders, 
    ACTIVITY_DIAGRAM_CLASSES
)


class OCRProcessor:
    """OCR processor for text extraction from images."""
    
    def __init__(self, use_angle_cls=True, lang='ch'):
        """
        Initialize OCR processor.
        
        Args:
            use_angle_cls: Whether to use angle classification
            lang: Language for OCR ('ch' for Chinese, 'en' for English)
        """
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.ocr = None
        
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr = PaddleOCR(
                    use_textline_orientation=use_angle_cls,
                    lang=lang
                )
                print(f"OCR initialized successfully (language: {lang})")
            except Exception as e:
                print(f"Failed to initialize OCR: {e}")
                self.ocr = None
        else:
            print("OCR not available - PaddleOCR not installed")
    
    def extract_text(self, image):
        if self.ocr is None:
            return []
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image is in RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Run OCR
            results = self.ocr.ocr(image)
            
            if not results or not results[0]:
                return []
            
            # Parse results - new PaddleOCR format
            ocr_results = []
            result = results[0]
            
            # Check if it's the new format (dictionary)
            if isinstance(result, dict):
                try:
                    # Extract text and scores
                    texts = result.get('rec_texts', [])
                    scores = result.get('rec_scores', [])
                    polys = result.get('rec_polys', [])
                    
                    for text, score, poly in zip(texts, scores, polys):
                        if text and text.strip():
                            # Convert polygon to bounding box
                            poly = np.array(poly)
                            x_coords = poly[:, 0]
                            y_coords = poly[:, 1]
                            x1, y1 = int(min(x_coords)), int(min(y_coords))
                            x2, y2 = int(max(x_coords)), int(max(y_coords))
                            
                            ocr_results.append({
                                'text': text.strip(),
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(score),
                                'polygon': poly.tolist()
                            })
                except Exception as e:
                    print(f"Warning: Failed to parse new OCR format: {e}")
            
            # Fallback for old format (list of lists)
            elif isinstance(result, list):
                for line in result:
                    if line and len(line) >= 2:
                        try:
                            # Extract bounding box and text
                            bbox = np.array(line[0])  # 4 points coordinates
                            text_info = line[1]
                            
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]  # text content
                                confidence = text_info[1]  # confidence score
                            else:
                                text = str(text_info)
                                confidence = 1.0
                            
                            # Convert 4-point bbox to [x1, y1, x2, y2] format
                            x_coords = bbox[:, 0]
                            y_coords = bbox[:, 1]
                            x1, y1 = int(min(x_coords)), int(min(y_coords))
                            x2, y2 = int(max(x_coords)), int(max(y_coords))
                            
                            ocr_results.append({
                                'text': text,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'polygon': bbox.tolist()
                            })
                        except Exception as e:
                            print(f"Warning: Failed to parse old OCR format: {e}")
                            continue
            
            return ocr_results
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return []


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def calculate_ap(recalls, precisions):
    # Add sentinel values
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    
    # Compute the precision envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def evaluate_predictions(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    num_classes = len(ACTIVITY_DIAGRAM_CLASSES) + 1
    
    # Per-class statistics
    class_stats = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0,
        'scores': [], 'matched': []
    })
    
    total_gt = 0
    total_pred = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        
        # Filter predictions by score threshold
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]
        
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        # Track which GT boxes have been matched
        gt_matched = [False] * len(gt_boxes)
        
        # Sort predictions by score (descending)
        sort_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sort_idx]
        pred_labels = pred_labels[sort_idx]
        pred_scores = pred_scores[sort_idx]
        
        # Match predictions to ground truth
        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT box with same class
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_label != pred_label:
                    continue
                if gt_matched[gt_idx]:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Record result
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                class_stats[pred_label]['tp'] += 1
                class_stats[pred_label]['scores'].append(pred_score)
                class_stats[pred_label]['matched'].append(True)
                gt_matched[best_gt_idx] = True
            else:
                class_stats[pred_label]['fp'] += 1
                class_stats[pred_label]['scores'].append(pred_score)
                class_stats[pred_label]['matched'].append(False)
        
        # Count false negatives (unmatched GT boxes)
        for gt_label, matched in zip(gt_labels, gt_matched):
            if not matched:
                class_stats[gt_label]['fn'] += 1
    
    # Calculate metrics
    metrics = {
        'per_class': {},
        'total_gt': total_gt,
        'total_pred': total_pred
    }
    
    all_aps = []
    
    for class_id in range(1, num_classes):  # Skip background
        stats = class_stats[class_id]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AP if we have predictions
        ap = 0
        if len(stats['scores']) > 0:
            scores = np.array(stats['scores'])
            matched = np.array(stats['matched'])
            
            # Sort by score
            sort_idx = np.argsort(-scores)
            matched = matched[sort_idx]
            
            # Calculate precision-recall curve
            tp_cumsum = np.cumsum(matched)
            fp_cumsum = np.cumsum(~matched)
            
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            recalls = tp_cumsum / (tp + fn) if (tp + fn) > 0 else tp_cumsum
            
            ap = calculate_ap(recalls, precisions)
        
        if tp + fn > 0:  # Only include classes that exist in GT
            all_aps.append(ap)
        
        class_name = ACTIVITY_DIAGRAM_CLASSES[class_id - 1] if class_id <= len(ACTIVITY_DIAGRAM_CLASSES) else f"class_{class_id}"
        
        metrics['per_class'][class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': tp + fn
        }
    
    # Calculate mAP
    metrics['mAP'] = np.mean(all_aps) if all_aps else 0
    
    # Overall precision/recall
    total_tp = sum(stats['tp'] for stats in class_stats.values())
    total_fp = sum(stats['fp'] for stats in class_stats.values())
    total_fn = sum(stats['fn'] for stats in class_stats.values())
    
    metrics['overall_precision'] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    metrics['overall_recall'] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    metrics['overall_f1'] = 2 * metrics['overall_precision'] * metrics['overall_recall'] / \
                           (metrics['overall_precision'] + metrics['overall_recall']) \
                           if (metrics['overall_precision'] + metrics['overall_recall']) > 0 else 0
    
    return metrics


@torch.no_grad()
def test_model(model, test_loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("\nRunning inference on test set...")
    for images, targets in tqdm(test_loader, desc="Testing"):
        images = [img.to(device) for img in images]
        
        # Get predictions
        predictions = model(images)
        
        # Store results
        all_predictions.extend([{k: v.cpu() for k, v in pred.items()} for pred in predictions])
        all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    
    print("\nCalculating metrics...")
    metrics = evaluate_predictions(all_predictions, all_targets, iou_threshold, score_threshold)
    
    return metrics, all_predictions, all_targets


def print_metrics(metrics, iou_threshold=0.5, score_threshold=0.5):
    """
    Print evaluation metrics in a nice format.
    """
    print("\n" + "="*80)
    print(f"TEST SET EVALUATION RESULTS")
    print(f"IoU Threshold: {iou_threshold} | Score Threshold: {score_threshold}")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  mAP@{iou_threshold:.2f}:        {metrics['mAP']:.4f}")
    print(f"  Precision:      {metrics['overall_precision']:.4f}")
    print(f"  Recall:         {metrics['overall_recall']:.4f}")
    print(f"  F1 Score:       {metrics['overall_f1']:.4f}")
    print(f"  Total GT boxes: {metrics['total_gt']}")
    print(f"  Total Pred:     {metrics['total_pred']}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<25} {'AP':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Support':<8}")
    print("-" * 80)
    
    # Sort by support (number of instances)
    sorted_classes = sorted(
        metrics['per_class'].items(),
        key=lambda x: x[1]['support'],
        reverse=True
    )
    
    for class_name, class_metrics in sorted_classes:
        if class_metrics['support'] > 0:  # Only show classes that exist
            print(f"{class_name:<25} "
                  f"{class_metrics['ap']:<8.4f} "
                  f"{class_metrics['precision']:<8.4f} "
                  f"{class_metrics['recall']:<8.4f} "
                  f"{class_metrics['f1']:<8.4f} "
                  f"{class_metrics['support']:<8}")
    
    print("="*80)


def visualize_predictions(images, predictions, targets, num_samples=5, 
                         score_threshold=0.5, save_dir='test_results', 
                         ocr_processor=None, ocr_confidence_threshold=0.7):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n📸 Saving visualizations to {save_dir}/")
    
    for idx in range(min(num_samples, len(images))):
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Get data
        image = images[idx]
        pred = predictions[idx]
        target = targets[idx]
        
        # Convert tensor to numpy
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        
        # Extract OCR text if processor is available
        ocr_results = []
        if ocr_processor is not None:
            try:
                # Convert numpy array to PIL Image for OCR
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                ocr_results = ocr_processor.extract_text(img_pil)
                print(f"Sample {idx+1}: Found {len(ocr_results)} text regions")
            except Exception as e:
                print(f"OCR failed for sample {idx+1}: {e}")
        
        # Ground Truth
        ax = axes[0]
        ax.imshow(img_np)
        ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        gt_boxes = target['boxes'].detach().cpu().numpy()
        gt_labels = target['labels'].detach().cpu().numpy()
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            
            class_name = ACTIVITY_DIAGRAM_CLASSES[label - 1] if label <= len(ACTIVITY_DIAGRAM_CLASSES) else f"class_{label}"
            ax.text(x1, y1 - 5, class_name, 
                   bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                   fontsize=8, color='white')
        
        # Predictions
        ax = axes[1]
        ax.imshow(img_np)
        ax.set_title(f'Predictions (threshold={score_threshold})', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        pred_boxes = pred['boxes'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
        pred_scores = pred['scores'].detach().cpu().numpy()
        
        # Filter by score
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            class_name = ACTIVITY_DIAGRAM_CLASSES[label - 1] if label <= len(ACTIVITY_DIAGRAM_CLASSES) else f"class_{label}"
            ax.text(x1, y1 - 5, f'{class_name} {score:.2f}',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                   fontsize=8, color='white')
        
        # Combined View (Predictions + OCR)
        ax = axes[2]
        ax.imshow(img_np)
        ax.set_title('Predictions + OCR Text', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Draw predictions
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            class_name = ACTIVITY_DIAGRAM_CLASSES[label - 1] if label <= len(ACTIVITY_DIAGRAM_CLASSES) else f"class_{label}"
            ax.text(x1, y1 - 5, f'{class_name} {score:.2f}',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                   fontsize=8, color='white')
        
        # Draw OCR text
        for ocr_result in ocr_results:
            if ocr_result['confidence'] >= ocr_confidence_threshold:
                x1, y1, x2, y2 = ocr_result['bbox']
                text = ocr_result['text']
                confidence = ocr_result['confidence']
                
                # Draw text bounding box
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
                )
                ax.add_patch(rect)
                
                # Add text label
                ax.text(x1, y2 + 10, f'{text} ({confidence:.2f})',
                       bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7),
                       fontsize=8, color='white')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'test_sample_{idx+1}.jpg', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {min(num_samples, len(images))} visualization(s)")


def save_metrics_report(metrics, save_path='test_results/metrics_report.txt', 
                       iou_threshold=0.5, score_threshold=0.5):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST SET EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  IoU Threshold:   {iou_threshold}\n")
        f.write(f"  Score Threshold: {score_threshold}\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  mAP@{iou_threshold:.2f}:        {metrics['mAP']:.4f}\n")
        f.write(f"  Precision:      {metrics['overall_precision']:.4f}\n")
        f.write(f"  Recall:         {metrics['overall_recall']:.4f}\n")
        f.write(f"  F1 Score:       {metrics['overall_f1']:.4f}\n")
        f.write(f"  Total GT boxes: {metrics['total_gt']}\n")
        f.write(f"  Total Pred:     {metrics['total_pred']}\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        f.write(f"{'Class':<25} {'AP':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Support':<8}\n")
        f.write("-" * 110 + "\n")
        
        sorted_classes = sorted(
            metrics['per_class'].items(),
            key=lambda x: x[1]['support'],
            reverse=True
        )
        
        for class_name, class_metrics in sorted_classes:
            if class_metrics['support'] > 0:
                f.write(f"{class_name:<25} "
                       f"{class_metrics['ap']:<8.4f} "
                       f"{class_metrics['precision']:<8.4f} "
                       f"{class_metrics['recall']:<8.4f} "
                       f"{class_metrics['f1']:<8.4f} "
                       f"{class_metrics['tp']:<6} "
                       f"{class_metrics['fp']:<6} "
                       f"{class_metrics['fn']:<6} "
                       f"{class_metrics['support']:<8}\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n💾 Saved detailed report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Faster R-CNN on Activity Diagrams Test Set with OCR')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for evaluation')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                       help='Score threshold for predictions')
    parser.add_argument('--num-vis', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--enable-ocr', action='store_true',
                       help='Enable OCR text extraction')
    parser.add_argument('--ocr-lang', type=str, default='ch',
                       help='OCR language (ch for Chinese, en for English)')
    parser.add_argument('--ocr-confidence', type=float, default=0.7,
                       help='OCR confidence threshold')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("\n" + "="*80)
    print("TESTING FASTER R-CNN ON ACTIVITY DIAGRAMS WITH OCR")
    print("="*80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output Directory: {args.output_dir}")
    print(f"OCR Enabled: {args.enable_ocr}")
    if args.enable_ocr:
        print(f"OCR Language: {args.ocr_lang}")
        print(f"OCR Confidence Threshold: {args.ocr_confidence}")
    
    # Initialize OCR processor
    ocr_processor = None
    if args.enable_ocr:
        print("\n🔤 Initializing OCR processor...")
        ocr_processor = OCRProcessor(
            use_angle_cls=True,
            lang=args.ocr_lang
        )
    
    # Load model
    print("\n📦 Loading model...")
    model = load_trained_model(args.checkpoint, device=device)
    
    # Load test data
    print("\n📂 Loading test dataset...")
    _, _, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=False  # No augmentation for testing
    )
    
    # Run evaluation
    metrics, predictions, targets = test_model(
        model, test_loader, device,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold
    )
    
    # Print results
    print_metrics(metrics, args.iou_threshold, args.score_threshold)
    
    # Save report
    save_metrics_report(
        metrics,
        save_path=f'{args.output_dir}/metrics_report.txt',
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold
    )
    
    # Visualize samples
    if args.num_vis > 0:
        # Get images from first batch
        images_to_vis = []
        preds_to_vis = []
        targets_to_vis = []
        
        for images, batch_targets in test_loader:
            batch_images = [img.to(device) for img in images]
            batch_preds = model(batch_images)
            
            images_to_vis.extend(images)
            preds_to_vis.extend([{k: v.detach().cpu() for k, v in p.items()} for p in batch_preds])
            targets_to_vis.extend(batch_targets)
            
            if len(images_to_vis) >= args.num_vis:
                break
        
        visualize_predictions(
            images_to_vis[:args.num_vis],
            preds_to_vis[:args.num_vis],
            targets_to_vis[:args.num_vis],
            num_samples=args.num_vis,
            score_threshold=args.score_threshold,
            save_dir=args.output_dir,
            ocr_processor=ocr_processor,
            ocr_confidence_threshold=args.ocr_confidence
        )
    
    print("\nTesting completed successfully!")
    print(f"All results saved to: {args.output_dir}/")
    print(f"\n Key Result: mAP@{args.iou_threshold} = {metrics['mAP']:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

