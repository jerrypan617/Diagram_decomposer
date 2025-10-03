# Activity Diagram Object Detection with OCR

This project implements a Faster R-CNN model for object detection in activity diagrams, enhanced with OCR text extraction capabilities.

## Features

- 🎯 **Object Detection**: Detects various elements in activity diagrams (nodes, edges, text boxes, etc.)
- 📝 **OCR Integration**: Extracts text from images using PaddleOCR
- 🎨 **Visualization**: Combined visualization of detection results and extracted text
- 📊 **Evaluation**: Comprehensive metrics and performance analysis

## Installation

1. Install Python dependencies:
```bash
pip install torch torchvision torchaudio
pip install paddlepaddle paddleocr
pip install matplotlib pillow opencv-python
pip install tqdm
```

2. Download the dataset (if needed):
```bash
python datas/data_downloader.py
```

## Usage

### Training

Train the object detection model:
```bash
python train_od.py --epochs 20 --batch-size 4
```

### Testing with OCR

Run inference with OCR text extraction:
```bash
python test_od.py --enable-ocr --num-vis 10
```

### Parameters

- `--enable-ocr`: Enable OCR text extraction
- `--ocr-lang`: OCR language (ch for Chinese, en for English)
- `--ocr-confidence`: OCR confidence threshold (0.0-1.0)
- `--num-vis`: Number of samples to visualize
- `--score-threshold`: Detection confidence threshold

## Output

The test script generates:
- **Metrics report**: Detailed performance metrics in `test_results/metrics_report.txt`
- **Visualizations**: Images showing ground truth, predictions, and OCR results
- **Combined view**: Detection boxes (red) + OCR text boxes (blue dashed)

## File Structure

```
source/
├── train_od.py          # Training script
├── test_od.py           # Testing script with OCR
├── models/
│   └── model.py         # Model definition
├── datas/
│   └── data_downloader.py # Dataset downloader
├── checkpoints/         # Model checkpoints (ignored by git)
├── test_results/        # Test outputs (ignored by git)
└── OCR_USAGE.md         # Detailed OCR usage guide
```

## Model Performance

The model achieves:
- **mAP@0.5**: ~0.49
- **Overall Precision**: ~0.71
- **Overall Recall**: ~0.89
- **F1 Score**: ~0.79

## OCR Integration

The OCR functionality:
- Uses PaddleOCR for text extraction
- Supports Chinese and English text
- Provides confidence scores for each detected text
- Combines with object detection results in visualization

See `OCR_USAGE.md` for detailed OCR usage instructions.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- PaddleOCR
- OpenCV
- Matplotlib
- PIL/Pillow

## License

This project is for research and educational purposes.
