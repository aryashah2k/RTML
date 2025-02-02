import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import cv2
from PIL import Image
import requests
from tqdm import tqdm
import json
import shutil
import random
from pathlib import Path
from pycocotools.coco import COCO
from utils.darknet import Darknet
from utils.yolo_loss import YOLOv4Loss
from utils.metrics import evaluate_coco_map
import argparse
from utils.util import *

class CustomTransform:
    def __call__(self, img):
        # Convert PIL Image to numpy array
        img = np.array(img)
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to 608x608
        img = cv2.resize(img, (608, 608))
        
        # Normalize and convert to tensor
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(img)

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def create_mini_dataset(val_dir, mini_val_dir, num_images=100):
    """Create a mini dataset from validation set"""
    os.makedirs(mini_val_dir, exist_ok=True)
    
    # Get list of all images
    all_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
    
    # Randomly select images
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    
    # Copy selected images
    for img_name in selected_images:
        src = os.path.join(val_dir, img_name)
        dst = os.path.join(mini_val_dir, img_name)
        shutil.copy2(src, dst)
    
    return [os.path.splitext(img)[0] for img in selected_images]

def filter_annotations(ann_file, mini_ann_file, selected_image_ids):
    """Filter annotations to only include selected images"""
    import json
    
    # Read original annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Convert image IDs to integers
    selected_image_ids = [int(img_id) for img_id in selected_image_ids]
    
    # Filter images
    annotations['images'] = [img for img in annotations['images'] 
                           if img['id'] in selected_image_ids]
    
    # Filter annotations
    annotations['annotations'] = [ann for ann in annotations['annotations'] 
                                if ann['image_id'] in selected_image_ids]
    
    # Save filtered annotations
    with open(mini_ann_file, 'w') as f:
        json.dump(annotations, f)

def setup_mini_coco(num_images=100):
    """Download and setup mini COCO dataset"""
    # Create directories
    data_dir = Path('data/mini_coco')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    val_dir = data_dir / 'val2017'
    mini_val_dir = data_dir / 'mini_val2017'
    ann_dir = data_dir / 'annotations'
    
    val_dir.mkdir(exist_ok=True)
    mini_val_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)
    
    # Download validation set if not exists
    val_zip = data_dir / 'val2017.zip'
    if not val_zip.exists():
        print("Downloading validation images...")
        download_file('http://images.cocodataset.org/zips/val2017.zip', str(val_zip))
    
    # Download annotations if not exists
    ann_zip = data_dir / 'annotations_trainval2017.zip'
    if not ann_zip.exists():
        print("Downloading annotations...")
        download_file('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', str(ann_zip))
    
    # Extract files if needed
    if not val_dir.exists() or not os.listdir(val_dir):
        print("Extracting validation images...")
        with zipfile.ZipFile(val_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    if not ann_dir.exists() or not os.listdir(ann_dir):
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Create mini dataset
    print(f"Creating mini dataset with {num_images} images...")
    selected_images = create_mini_dataset(val_dir, mini_val_dir, num_images)
    
    # Filter annotations
    print("Filtering annotations...")
    filter_annotations(
        ann_dir / 'instances_val2017.json',
        ann_dir / 'instances_mini_val2017.json',
        selected_images
    )
    
    # Create mini COCO config
    mini_coco_config = {
        'train': str(mini_val_dir),  # Using val set for both train and val in mini version
        'train_annotations': str(ann_dir / 'instances_mini_val2017.json'),
        'val': str(mini_val_dir),
        'val_annotations': str(ann_dir / 'instances_mini_val2017.json'),
        'nc': 80,
        'names': 'data/coco.names'
    }
    
    with open(data_dir / 'mini_coco.yaml', 'w') as f:
        yaml.dump(mini_coco_config, f)
    
    # Clean up zip files
    if val_zip.exists():
        val_zip.unlink()
    if ann_zip.exists():
        ann_zip.unlink()
    
    print("\nMini COCO dataset setup complete!")
    print(f"- Images: {mini_val_dir}")
    print(f"- Annotations: {ann_dir / 'instances_mini_val2017.json'}")
    print(f"- Config: {data_dir / 'mini_coco.yaml'}")

def get_coco_label_map():
    """Get mapping of COCO category IDs to consecutive class indices (0-79)"""
    # COCO class labels (80 classes)
    coco_labels = {
        1: 0,   # person
        2: 1,   # bicycle
        3: 2,   # car
        4: 3,   # motorcycle
        5: 4,   # airplane
        6: 5,   # bus
        7: 6,   # train
        8: 7,   # truck
        9: 8,   # boat
        10: 9,  # traffic light
        11: 10, # fire hydrant
        13: 11, # stop sign
        14: 12, # parking meter
        15: 13, # bench
        16: 14, # bird
        17: 15, # cat
        18: 16, # dog
        19: 17, # horse
        20: 18, # sheep
        21: 19, # cow
        22: 20, # elephant
        23: 21, # bear
        24: 22, # zebra
        25: 23, # giraffe
        27: 24, # backpack
        28: 25, # umbrella
        31: 26, # handbag
        32: 27, # tie
        33: 28, # suitcase
        34: 29, # frisbee
        35: 30, # skis
        36: 31, # snowboard
        37: 32, # sports ball
        38: 33, # kite
        39: 34, # baseball bat
        40: 35, # baseball glove
        41: 36, # skateboard
        42: 37, # surfboard
        43: 38, # tennis racket
        44: 39, # bottle
        46: 40, # wine glass
        47: 41, # cup
        48: 42, # fork
        49: 43, # knife
        50: 44, # spoon
        51: 45, # bowl
        52: 46, # banana
        53: 47, # apple
        54: 48, # sandwich
        55: 49, # orange
        56: 50, # broccoli
        57: 51, # carrot
        58: 52, # hot dog
        59: 53, # pizza
        60: 54, # donut
        61: 55, # cake
        62: 56, # chair
        63: 57, # couch
        64: 58, # potted plant
        65: 59, # bed
        67: 60, # dining table
        70: 61, # toilet
        72: 62, # tv
        73: 63, # laptop
        74: 64, # mouse
        75: 65, # remote
        76: 66, # keyboard
        77: 67, # cell phone
        78: 68, # microwave
        79: 69, # oven
        80: 70, # toaster
        81: 71, # sink
        82: 72, # refrigerator
        84: 73, # book
        85: 74, # clock
        86: 75, # vase
        87: 76, # scissors
        88: 77, # teddy bear
        89: 78, # hair drier
        90: 79  # toothbrush
    }
    return coco_labels

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        Args:
            root: Path to image directory
            annFile: Path to annotation file
            transform: Optional transform to be applied on a sample
        """
        self.root = root
        self.transform = transform
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Create category mapping
        self.cat_mapping = {}
        for cat_id in self.coco.cats.keys():
            self.cat_mapping[cat_id] = len(self.cat_mapping)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary containing:
                  'boxes', 'labels', 'image_id', 'orig_size'
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            # Skip crowd annotations
            if ann.get('iscrowd', 0):
                continue
                
            # Get bbox in [x1, y1, x2, y2] format
            x, y, w, h = ann['bbox']
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
                
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_mapping[ann['category_id']])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.as_tensor([orig_size[1], orig_size[0]])  # H, W format
        }
        
        return img, target
    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable number of objects
        Args:
            batch: List of tuples (image, target)
        Returns:
            images: Tensor of shape (batch_size, C, H, W)
            targets: List of target dictionaries
        """
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        images = torch.stack(images, 0)
        return images, targets

def convert_targets(targets):
    """Convert COCO format targets to YOLOv4 format"""
    boxes = []
    labels = []
    
    for target in targets:
        bbox = target['bbox']  # [x, y, width, height]
        # Convert to [x1, y1, x2, y2]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        
        # Normalize coordinates
        x1, x2 = x1/608, x2/608
        y1, y2 = y1/608, y2/608
        
        boxes.append([x1, y1, x2, y2])
        labels.append(target['category_id'])
    
    if len(boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64)
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    return boxes_tensor, labels_tensor

def compute_loss(predictions, targets, device):
    """Compute YOLOv4 loss"""
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        pred = predictions[i].view(-1, 85)  # 85 = 4 (box coords) + 1 (objectness) + 80 (class scores)
        pred_boxes = pred[:, :4]
        
        target_boxes = targets[i]['boxes']
        if len(target_boxes) > 0:
            # Create prediction tensor that requires gradients
            pred_boxes = pred_boxes[:len(target_boxes)].clone()
            pred_boxes.requires_grad_(True)
            
            # Compute box coordinate loss
            box_loss = F.mse_loss(pred_boxes, target_boxes, reduction='sum')
            
            # Add to total loss
            total_loss = total_loss + box_loss
    
    # Normalize loss by batch size
    total_loss = total_loss / batch_size
    return total_loss

def train_on_mini_coco(num_epochs=10):
    """Train YOLOv4 on mini COCO dataset"""
    print("\nTraining on mini COCO dataset...")
    
    # Load model
    print("Loading pretrained weights...")
    model = Darknet('cfg/yolov4.cfg')
    if os.path.exists('weights/yolov4.weights'):
        model.load_weights('weights/yolov4.weights')
    else:
        print("Warning: No pretrained weights found at weights/yolov4.weights")
        print("Training from scratch...")
    
    # Create output directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load mini COCO config
    with open('data/mini_coco/mini_coco.yaml', 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # Create dataset and dataloader
    print(f"\nCreating dataset from: {data_dict['train']}")
    print(f"Using annotations: {data_dict['train_annotations']}")
    
    train_dataset = COCODataset(
        data_dict['train'],
        data_dict['train_annotations'],
        transform=CustomTransform()
    )
    
    val_dataset = COCODataset(
        data_dict['val'],
        data_dict['val_annotations'],
        transform=CustomTransform()
    )
    
    print(f"Dataset size: {len(train_dataset)} training images, {len(val_dataset)} validation images")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=val_dataset.collate_fn
    )
    
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = YOLOv4Loss(model.anchors, num_classes=80)
    
    # Load COCO ground truth for evaluation
    coco_gt = COCO(data_dict['val_annotations'])
    
    print("\nStarting training...")
    best_map = 0.0
    
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            
            for batch_i, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(device)
                
                # Move target tensors to device
                for t in targets:
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            t[k] = v.to(device)
                
                # Forward pass
                predictions = model(imgs, device == torch.device("cuda"))
                
                # Calculate loss
                loss = criterion(predictions, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update running loss
                epoch_loss += loss.item()
                
                if batch_i % 10 == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_i}, Loss: {epoch_loss/(batch_i+1):.4f}")
            
            # Evaluate mAP on validation set
            print("\nEvaluating mAP...")
            model.eval()
            current_map = evaluate_coco_map(model, val_loader, coco_gt)
            print(f"Epoch {epoch} mAP: {current_map:.4f}")
            
            # Save checkpoint if mAP improved
            if current_map > best_map:
                best_map = current_map
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mAP': current_map,
                }
                torch.save(checkpoint, f'checkpoints/yolov4_best.pth')
                print(f"Saved new best model with mAP: {current_map:.4f}")
        
        print("\nTraining complete!")
        print(f"Best mAP: {best_map:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return model

def detect(model, image_path, conf_thres=0.5, nms_thres=0.4):
    """
    Detect objects in an image
    Args:
        model: YOLOv4 model
        image_path: Path to image file
        conf_thres: Confidence threshold
        nms_thres: Non-maximum suppression threshold
    Returns:
        List of detections, on (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    print("Starting detection...")
    model.eval()  # Set in evaluation mode
    
    # Read and preprocess image
    print("Reading image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return []
        
    print("Converting color space...")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    orig_h, orig_w = img.shape[:2]
    print(f"Original image size: {orig_w}x{orig_h}")
    
    # Resize and prepare image
    print("Preprocessing image...")
    img = cv2.resize(img, (608, 608))
    img = img.transpose(2, 0, 1)  # Convert to channel-first
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img = img / 255.0  # Normalize
    
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # Add batch dimension
    
    # Get detections
    print("Running model inference...")
    with torch.no_grad():
        img = img.cuda()
        try:
            detections = model(img, True)  # Forward pass
            print(f"Raw detections shape: {detections.shape}")
            
            # Apply NMS
            print("Applying NMS...")
            detections = non_max_suppression(detections, conf_thres, nms_thres)
            print(f"After NMS: {len(detections)} images with detections")
            
            if detections[0] is not None:
                # Rescale boxes to original image
                detections = detections[0]
                print(f"Number of detections: {len(detections)}")
                detections[:, :4] = scale_coords(detections[:, :4], (orig_h, orig_w), (608, 608))
            else:
                print("No detections after NMS")
                detections = []
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            detections = []
    
    return detections

def detect_on_mini_coco(model, conf_thres=0.5, nms_thres=0.4):
    """Run detection on mini COCO validation set"""
    print("\nInitializing detection...")
    if model is None:
        print("No model provided. Loading from weights...")
        model = Darknet('cfg/yolov4.cfg')
        if os.path.exists('weights/yolov4.weights'):
            print("Loading YOLOv4 weights...")
            model.load_weights('weights/yolov4.weights')
        else:
            print("No weights found at weights/yolov4.weights")
            return
    
    print("Moving model to GPU...")
    model.cuda()
    model.eval()
    
    # Load mini COCO config
    print("Loading mini COCO config...")
    with open('data/mini_coco/mini_coco.yaml', 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # Get validation image directory
    val_dir = data_dict['val']
    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}")
        return
    
    print(f"Using validation directory: {val_dir}")
    
    # Create output directory for visualization
    output_dir = 'output/detections'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    # Get COCO class names
    coco_names = list(get_coco_label_map().values())
    
    # Process each image
    image_files = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nFound {len(image_files)} images to process")
    
    for img_name in image_files:
        image_path = os.path.join(val_dir, img_name)
        print(f"\nProcessing {img_name}...")
        
        try:
            # Get detections
            detections = detect(model, image_path, conf_thres, nms_thres)
            
            # Load and draw on image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image for visualization: {image_path}")
                continue
            
            if len(detections) > 0:
                print(f"Drawing {len(detections)} detections...")
                for x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in detections:
                    # Convert to integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cls_pred = int(cls_pred)
                    
                    if cls_pred >= len(coco_names):
                        print(f"Warning: Invalid class prediction {cls_pred}")
                        continue
                    
                    # Draw box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f'{coco_names[cls_pred]} {obj_conf*cls_conf:.2f}'
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.putText(img, label, (x1, y1 + t_size[1] + 4), 
                               cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
            
            # Save output image
            output_path = os.path.join(output_dir, f'detected_{img_name}')
            cv2.imwrite(output_path, img)
            print(f"Saved detection result to {output_path}")
            print(f"Found {len(detections)} objects")
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    print("Starting NMS...")
    
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None] * prediction.shape[0]
    
    for image_i, image_pred in enumerate(prediction):
        print(f"Processing image {image_i + 1}/{prediction.shape[0]}")
        
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
            
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            
            # Update weights
            weights = detections[invalid, 4:5]
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
            
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
            print(f"Kept {len(keep_boxes)} boxes after NMS")
    
    return output

def xywh2xyxy(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    """
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = x - w/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = y - h/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = x + w/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = y + h/2
    return y

def scale_coords(coords, orig_shape, new_shape):
    """
    Rescale coords (x1, y1, x2, y2) from new_shape to original shape
    """
    gain = min(new_shape[0] / orig_shape[0], new_shape[1] / orig_shape[1])  # gain  = old / new
    pad = (new_shape[1] - orig_shape[1] * gain) / 2, (new_shape[0] - orig_shape[0] * gain) / 2  # wh padding
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, 0].clamp_(0, orig_shape[1])  # x1
    coords[:, 1].clamp_(0, orig_shape[0])  # y1
    coords[:, 2].clamp_(0, orig_shape[1])  # x2
    coords[:, 3].clamp_(0, orig_shape[0])  # y2
    
    return coords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', action='store_true', help='Setup mini COCO dataset')
    parser.add_argument('--train', action='store_true', help='Train on mini COCO')
    parser.add_argument('--detect', action='store_true', help='Run detection on mini COCO')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images for mini dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    
    if args.setup:
        setup_mini_coco(args.num_images)
    
    if args.train or args.detect:
        if not os.path.exists('data/mini_coco/mini_coco.yaml'):
            print("Mini COCO dataset not found! Run with --setup first.")
            return
        
        model = None
        if args.train:
            print("\nTraining on mini COCO dataset...")
            model = train_on_mini_coco(args.epochs)
        
        if args.detect:
            if model is None:
                model = Darknet('cfg/yolov4.cfg')
                if os.path.exists('weights/yolov4.weights'):
                    model.load_weights('weights/yolov4.weights')
            
            print("\nRunning detection on mini COCO dataset...")
            detect_on_mini_coco(model)

if __name__ == '__main__':
    main()
