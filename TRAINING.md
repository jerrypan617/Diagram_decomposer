# Training Faster R-CNN on Activity Diagrams

## Quick Start

### Basic Training
```bash
uv run python source/train_od.py
```

### Custom Parameters
```bash
uv run python source/train_od.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 0.005 \
    --num-workers 4 \
    --checkpoint-dir checkpoints
```

## Training Parameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 4)
- `--lr`: Initial learning rate (default: 0.005)
- `--num-workers`: Number of data loading workers (default: 4)
- `--checkpoint-dir`: Directory to save model checkpoints (default: 'checkpoints')

## What the Script Does

1. **Loads Dataset**: Downloads and prepares the activity-diagrams-qdobr dataset
   - Train: 259 samples
   - Validation: 45 samples
   - Test: Available for final evaluation

2. **Model Setup**: 
   - Initializes Faster R-CNN with ResNet50-FPN backbone
   - Configures for 21 classes (20 diagram elements + background)
   - Uses pretrained COCO weights for transfer learning

3. **Training Loop**:
   - SGD optimizer with momentum (0.9)
   - Learning rate scheduler (step decay every 10 epochs)
   - Progress bars with loss tracking
   - Automatic validation after each epoch

4. **Checkpointing**:
   - `best_model.pth`: Best model based on validation loss
   - `checkpoint_epoch_X.pth`: Saved every 10 epochs
   - `final_model.pth`: Model after final epoch

## Dataset Classes (20 categories)

The model will learn to detect:
- **Control Flow**: control_flow, object_flow
- **Nodes**: start_node, final_node, decision_node, exit_node, fork, merge
- **Activities**: action, activity, activity-diagrams
- **Objects**: object
- **Signals**: signal_send, signal_recept
- **Text**: text, commeent (comment)
- **Others**: final_flow_node, merge_noode

## Expected Training Time

- **CPU**: ~15-30 minutes per epoch (not recommended)
- **GPU (CUDA)**: ~2-5 minutes per epoch
- **Full training (50 epochs)**: 2-4 hours on GPU

## Using Trained Model

### Load and Use
```python
from source.train_od import load_trained_model

# Load best model
model = load_trained_model('checkpoints/best_model.pth')

# Use for inference
model.eval()
# ... your inference code
```

### Inference on New Images
```python
import torch
from PIL import Image
import torchvision.transforms as T

# Load image
image = Image.open('diagram.jpg')
image_tensor = T.ToTensor()(image)

# Run inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

with torch.no_grad():
    predictions = model([image_tensor.to(device)])

# Access results
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']
```

## Tips for Better Training

1. **Start with default parameters** - They're tuned for this dataset
2. **Monitor validation loss** - Should decrease steadily
3. **Use GPU if available** - Much faster training
4. **Adjust batch size** based on your GPU memory:
   - 8GB GPU: batch_size=2-4
   - 16GB GPU: batch_size=4-8
   - 24GB+ GPU: batch_size=8-16

## Troubleshooting

### Out of Memory
Reduce batch size: `--batch-size 2` or `--batch-size 1`

### Training Too Slow
- Reduce `--num-workers` if CPU bottleneck
- Use smaller image sizes (requires code modification)

### Poor Performance
- Train for more epochs
- Adjust learning rate
- Check data augmentation settings

## Next Steps

After training:
1. Evaluate on test set
2. Visualize predictions on validation samples
3. Fine-tune hyperparameters if needed
4. Export model for deployment

