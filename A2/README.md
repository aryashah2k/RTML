# YOLOv4 with CIoU Loss Implementation

This project implements YOLOv4 object detection with Complete IoU (CIoU) loss function and evaluates its performance on the COCO dataset. The implementation includes both training and inference capabilities, with a focus on improved bounding box regression through CIoU loss.

## Project Structure

```
YOLOv4_Project/
├── cfg/
│   └── yolov4.cfg           # YOLOv4 model configuration
├── data/
│   ├── coco/               # COCO dataset directory
│   ├── coco.names         # COCO class names
│   └── coco.yaml          # Dataset configuration
├── utils/
│   ├── darknet.py         # Darknet model architecture
│   ├── metrics.py         # Evaluation metrics (mAP)
│   ├── mish.py           # Mish activation function
│   ├── util.py           # Utility functions
│   └── yolo_loss.py      # YOLOv4 loss with CIoU
├── weights/
│   ├── csdarknet53-omega_final.weights
│   └── yolov4.weights    # Pretrained weights
├── main.py               # Main training/inference script
├── mini_coco.py         # Mini COCO dataset testing
├── download_coco.py     # COCO dataset downloader
└── requirements.txt     # Project dependencies
```

## Key Components and Changes

### Core Scripts

#### 1. main.py
- Main script for training and inference
- Features:
  - Training with CIoU loss
  - Real-time mAP evaluation
  - Model checkpointing
  - Inference with visualization
  - Results saved in dedicated 'results' directory

#### 2. mini_coco.py
- Testing script using a smaller subset of COCO
- Implements:
  - Dataset creation and handling
  - Training loop with CIoU loss
  - Evaluation metrics
  - Serves as a prototype for main.py

### Utility Modules

#### 1. utils/yolo_loss.py
- Implements YOLOv4 loss function with CIoU
- Key features:
  - Complete IoU calculation
  - Aspect ratio consistency
  - Center point distance
  - Better bounding box regression

#### 2. utils/metrics.py
- Evaluation metrics implementation
- Features:
  - Mean Average Precision (mAP) calculation
  - COCO evaluation metrics
  - Detection result processing

#### 3. utils/darknet.py
- YOLOv4 model architecture
- Includes:
  - CSPDarknet53 backbone
  - SPP module
  - PANet neck
  - YOLOv4 head

#### 4. utils/util.py
- Utility functions for:
  - Bounding box operations
  - Non-maximum suppression
  - Result processing
  - Data transformation

## Assignment Requirements and Implementation

### 1. CIoU Loss Implementation
- Implemented in `utils/yolo_loss.py`
- Features:
  - IoU calculation
  - Center point distance term
  - Aspect ratio consistency term
  - Complete loss formulation

### 2. COCO Dataset Integration
- Full COCO dataset support
- Custom dataset loader
- Evaluation using COCO metrics
- Mini COCO subset for testing

### 3. Training and Evaluation
- Training:
  - Batch processing
  - Learning rate scheduling
  - Loss computation with CIoU
  - Model checkpointing
- Evaluation:
  - Real-time mAP calculation
  - COCO evaluation metrics
  - Performance tracking

### 4. Visualization
- Detection visualization with:
  - Bounding boxes
  - Class labels
  - Confidence scores
  - Results saved in 'results' directory

### 5. Inference Results

|Script used for inference|Output|
|-------------------------|------|
|mini_coco.py|!(1)[]|
|mini_coco.py|!(2)[]|
|mini_coco.py|!(3)[]|
|main.py|!(4)[]|


## Usage

### Training
```bash
python main.py --data data/coco.yaml --epochs 100 --batch-size 4
```

### Inference
```bash
python main.py --img path/to/image.jpg --weights weights/yolov4.weights
```

### Testing with Mini COCO
```bash
python mini_coco.py
```

## Dependencies
See `requirements.txt` for the complete list of dependencies.

## Results
Detection results are saved in the 'results' directory with the format:
`[original_image_name]_detection.jpg`

## Note
This implementation focuses on improving object detection accuracy through the CIoU loss function while maintaining the original YOLOv4 architecture. The addition of CIoU loss helps in better bounding box regression, particularly for objects with varying aspect ratios and sizes.
