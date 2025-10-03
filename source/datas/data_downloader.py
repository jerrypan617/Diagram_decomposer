import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def load_activity_diagrams_dataset():
    """
    Load the activity-diagrams-qdobr dataset.
    
    Returns:
        dataset: HuggingFace dataset object
    """
    print("Loading activity-diagrams-qdobr dataset...")
    ds = load_dataset("Francesco/activity-diagrams-qdobr")
    print(f"Dataset loaded successfully!")
    print(f"Splits available: {list(ds.keys())}")
    return ds


def explore_dataset(dataset, split='train'):
    """
    Explore the structure of the dataset.
    
    Args:
        dataset: HuggingFace dataset object
        split: Dataset split to explore
    """
    print(f"\n=== Exploring {split} split ===")
    split_data = dataset[split]
    print(f"Number of samples: {len(split_data)}")
    
    # Get first sample
    sample = split_data[0]
    print(f"\nSample keys: {sample.keys()}")
    
    # Explore image
    if 'image' in sample:
        image = sample['image']
        print(f"Image type: {type(image)}")
        if hasattr(image, 'size'):
            print(f"Image size: {image.size}")
    
    # Explore annotations
    if 'objects' in sample:
        print(f"\nAnnotation structure:")
        objects = sample['objects']
        print(f"Objects keys: {objects.keys() if hasattr(objects, 'keys') else type(objects)}")
        if isinstance(objects, dict):
            for key, value in objects.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: {len(value)} items, first item: {value[0]}")
                else:
                    print(f"  {key}: {value}")
    
    return sample


def visualize_sample(sample, save_path='sample_visualization.jpg', title='Activity Diagram Detection'):
    """
    Visualize a sample with bounding boxes and labels.
    
    Args:
        sample: A sample from the dataset
        save_path: Path to save the visualization
        title: Title for the plot
    """
    # Extract image
    image = sample['image']
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img_array)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Extract annotations
    if 'objects' in sample:
        objects = sample['objects']
        
        # Handle different annotation formats
        if isinstance(objects, dict):
            # Extract bounding boxes and labels
            bboxes = objects.get('bbox', [])
            labels = objects.get('category', [])
            
            # Generate random colors for each category
            unique_labels = list(set(labels)) if labels else []
            np.random.seed(42)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            # Draw each bounding box
            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                # bbox format might be [x, y, width, height] or [x_min, y_min, x_max, y_max]
                if len(bbox) == 4:
                    # Assume COCO format: [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Get color for this label
                    color = color_map.get(label, 'red')
                    
                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label text
                    ax.text(
                        x, y - 5,
                        f"{label}",
                        color='white',
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7)
                    )
            
            print(f"\nVisualized {len(bboxes)} objects")
            print(f"Categories found: {unique_labels}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    # Also display if in interactive mode
    try:
        plt.show()
    except:
        pass
    
    plt.close()


def visualize_validation_sample(dataset, index=0, save_path='validation_sample.jpg'):
    """
    Load and visualize a sample from the validation set.
    
    Args:
        dataset: HuggingFace dataset object
        index: Index of the sample to visualize
        save_path: Path to save the visualization
    """
    # Get validation split
    if 'validation' in dataset:
        validation_data = dataset['validation']
    elif 'val' in dataset:
        validation_data = dataset['val']
    elif 'test' in dataset:
        print("No validation split found, using test split instead")
        validation_data = dataset['test']
    else:
        print("No validation/test split found, using train split")
        validation_data = dataset['train']
    
    print(f"\nLoading sample {index} from validation set ({len(validation_data)} total samples)...")
    sample = validation_data[index]
    
    # Visualize
    visualize_sample(sample, save_path=save_path, title=f'Validation Sample {index}')
    
    return sample


if __name__ == "__main__":
    # Load dataset
    ds = load_activity_diagrams_dataset()
    
    # Explore dataset structure
    sample = explore_dataset(ds, split='train')
    
    # Visualize a validation sample
    print("\n" + "="*60)
    print("Visualizing validation sample...")
    print("="*60)
    validation_sample = visualize_validation_sample(ds, index=0)
    
    print("\n🎉 Done! Check 'validation_sample.jpg' for the visualization.")