import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
from utils.darknet import Darknet
from utils.util import *
from utils.yolo_loss import YOLOv4Loss
from utils.metrics import evaluate_coco_map
from pycocotools.coco import COCO
import argparse
import os
import yaml
from PIL import Image

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

class COCODataset:
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
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_mapping[ann['category_id']])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor(orig_size)
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

def prepare_image(img_path, img_size=608):
    # Check if image exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :] / 255.0
    img = torch.tensor(img, dtype=torch.float)
    return img

def train_yolo(model, train_loader, val_loader, coco_gt, device, num_epochs=100):
    """Train YOLOv4 with CIoU loss and mAP evaluation"""
    print("\nStarting training...")
    
    # Initialize model, optimizer, and loss function
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = YOLOv4Loss(model.anchors, num_classes=80)
    
    # Create output directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
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

def draw_detections(image, detections, class_names=None):
    """Draw bounding boxes and labels on the image"""
    img = image.copy()
    
    # Generate random colors for each class
    np.random.seed(42)  # for consistent colors
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
    for detection in detections:
        # Get detection info
        _, x1, y1, x2, y2, obj_conf, cls_conf, cls_pred = detection
        cls_pred = int(cls_pred)
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Get color for this class
        color = tuple(map(int, colors[cls_pred % len(colors)]))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if class_names and cls_pred < len(class_names):
            label = f"{class_names[cls_pred]} {cls_conf:.2f}"
        else:
            label = f"Class {cls_pred} {cls_conf:.2f}"
        
        # Get label size for background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            img,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1,  # filled
        )
        
        # Draw label text
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='path to model config file')
    parser.add_argument('--weights', type=str, default='weights/yolov4.weights', help='path to weights file')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='path to data config file')
    parser.add_argument('--img', type=str, default='', help='path to input image')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = Darknet(args.cfg)
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        model.load_weights(args.weights)
    else:
        print("Warning: No pretrained weights found")
        print("Training from scratch...")
    
    if args.img:
        # Inference mode
        # Convert relative path to absolute path if needed
        img_path = os.path.abspath(args.img)
        if not os.path.exists(img_path):
            # Try relative to the current directory
            img_path = os.path.join(os.getcwd(), args.img)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {args.img}")
        
        print(f"Loading image: {img_path}")
        
        # Load original image for visualization
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for model
        model.eval()
        img = prepare_image(img_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = img.to(device)
        model = model.to(device)
        
        with torch.no_grad():
            predictions = model(img, device == torch.device("cuda"))
            detections = write_results(predictions, confidence=args.conf_thres, num_classes=80, nms_conf=args.nms_thres)
        
        if detections is not None:
            # Convert detections to numpy for easier handling
            detections = detections.cpu().numpy()
            print(f"Found {len(detections)} objects!")
            
            # Load COCO class names if available
            try:
                with open('data/coco.names', 'r') as f:
                    class_names = f.read().strip().split('\n')
            except:
                class_names = None
            
            # Print detections
            for detection in detections:
                # Detection format: [batch_idx, x1, y1, x2, y2, obj_conf, cls_conf, cls_pred]
                _, x1, y1, x2, y2, obj_conf, cls_conf, cls_pred = detection
                cls_pred = int(cls_pred)
                
                # Print class name if available, otherwise just the class index
                if class_names and cls_pred < len(class_names):
                    class_name = class_names[cls_pred]
                    print(f"Class: {class_name} ({cls_pred}), Confidence: {cls_conf:.4f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                else:
                    print(f"Class: {cls_pred}, Confidence: {cls_conf:.4f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            # Draw detections on image
            result_img = draw_detections(orig_img, detections, class_names)
            
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.getcwd(), 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate output filename using original image name
            orig_filename = os.path.basename(img_path)
            filename_without_ext = os.path.splitext(orig_filename)[0]
            output_path = os.path.join(results_dir, f'{filename_without_ext}_detection.jpg')
            
            # Save result
            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"\nDetection result saved to: {output_path}")
    else:
        # Training mode
        print("Loading data configuration...")
        with open(args.data, 'r') as f:
            data_dict = yaml.safe_load(f)
        
        # Setup datasets
        print(f"\nCreating datasets...")
        print(f"Training data: {data_dict['train']}")
        print(f"Training annotations: {data_dict['train_annotations']}")
        print(f"Validation data: {data_dict['val']}")
        print(f"Validation annotations: {data_dict['val_annotations']}")
        
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
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=COCODataset.collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=COCODataset.collate_fn
        )
        
        # Load COCO ground truth for evaluation
        coco_gt = COCO(data_dict['val_annotations'])
        
        # Train model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_yolo(model, train_loader, val_loader, coco_gt, device, num_epochs=args.epochs)

if __name__ == '__main__':
    main()