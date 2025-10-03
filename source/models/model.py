import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np


def load_pretrained_faster_rcnn(num_classes=None, pretrained=True):
    """
    Load a pre-trained Faster R-CNN model.
    
    Args:
        num_classes: Number of classes for detection (including background).
                    If None, uses the default COCO pretrained model (91 classes).
        pretrained: Whether to load pretrained weights.
    
    Returns:
        model: Faster R-CNN model
    """
    if pretrained and num_classes is None:
        # Load model with COCO pretrained weights (91 classes)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()  # Set to evaluation mode
        return model
    
    elif pretrained and num_classes is not None:
        # Load pretrained backbone, but modify classifier for custom num_classes
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        
        # Replace the classifier head with a new one for your num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        return model
    
    else:
        # Load model without pretrained weights
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes or 91)
        return model


# COCO class names (91 classes including background at index 0)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        image: PIL Image
    """
    image = Image.open(image_path).convert("RGB")
    return image


def preprocess_image(image):
    """
    Preprocess image for Faster R-CNN model.
    
    Args:
        image: PIL Image
    
    Returns:
        tensor: Preprocessed image tensor
    """
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(image)


def detect_objects(model, image_path, confidence_threshold=0.5, device='cpu'):
    """
    Detect objects in an image using Faster R-CNN.
    
    Args:
        model: Faster R-CNN model
        image_path: Path to input image
        confidence_threshold: Minimum confidence score for detections
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        image: Original PIL Image
        predictions: Dictionary with boxes, labels, and scores
    """
    # Load and preprocess image
    image = load_image(image_path)
    image_tensor = preprocess_image(image).to(device)
    
    # Run inference
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    
    # Filter by confidence threshold
    mask = predictions['scores'] >= confidence_threshold
    filtered_predictions = {
        'boxes': predictions['boxes'][mask].cpu(),
        'labels': predictions['labels'][mask].cpu(),
        'scores': predictions['scores'][mask].cpu()
    }
    
    return image, filtered_predictions


def visualize_detections(image, predictions, output_path='output.jpg', class_names=None):
    """
    Visualize object detection results on the image.
    
    Args:
        image: PIL Image
        predictions: Dictionary with boxes, labels, and scores
        output_path: Path to save the output image
        class_names: List of class names (defaults to COCO classes)
    """
    if class_names is None:
        class_names = COCO_CLASSES
    
    # Create a copy of the image to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw each detection
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    
    # Generate colors for each class
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
              for _ in range(len(class_names))]
    
    for box, label, score in zip(boxes, labels, scores):
        # Get box coordinates
        x1, y1, x2, y2 = box.tolist()
        
        # Get class name and color
        class_idx = label.item()
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        color = colors[class_idx % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background and text
        label_text = f"{class_name}: {score:.2f}"
        
        # Get text bounding box for background
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1), label_text, fill='white', font=font)
    
    # Save the image
    img_draw.save(output_path)
    print(f"Saved detection results to {output_path}")
    print(f"Detected {len(boxes)} objects")
    
    return img_draw


def detect_and_visualize(image_path, output_path='output.jpg', confidence_threshold=0.5, device='cpu'):
    """
    Complete pipeline: load model, detect objects, and visualize results.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        confidence_threshold: Minimum confidence score for detections
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        result_image: PIL Image with detections drawn
    """
    # Load model
    print("Loading Faster R-CNN model...")
    model = load_pretrained_faster_rcnn()
    
    # Detect objects
    print(f"Detecting objects in {image_path}...")
    image, predictions = detect_objects(model, image_path, confidence_threshold, device)
    
    # Visualize results
    print("Visualizing detections...")
    result_image = visualize_detections(image, predictions, output_path)
    
    # Print detection details
    print("\nDetection details:")
    for i, (label, score) in enumerate(zip(predictions['labels'], predictions['scores'])):
        class_name = COCO_CLASSES[label.item()]
        print(f"  {i+1}. {class_name}: {score:.3f}")
    
    return result_image


# Example usage:
if __name__ == "__main__":
    import sys
    
    # Quick start: Detect objects in in.jpg and save to output.jpg
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
        output_image = sys.argv[2] if len(sys.argv) > 2 else 'output.jpg'
        confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    else:
        input_image = 'in.jpg'
        output_image = 'output.jpg'
        confidence = 0.5
    
    # Run detection and visualization
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        result = detect_and_visualize(
            image_path=input_image,
            output_path=output_image,
            confidence_threshold=confidence,
            device=device
        )
        print(f"\n✅ Done! Check {output_image} for results.")
    except FileNotFoundError:
        print(f"Error: Could not find image '{input_image}'")
        print("\nUsage:")
        print(f"  python {__file__} <input_image> [output_image] [confidence_threshold]")
        print(f"\nExample:")
        print(f"  python {__file__} in.jpg output.jpg 0.5")

