import numpy as np
import torch
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves."""
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Create a list of indexes where the recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Calculate the area under PR curve by sum of rectangular blocks
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def bbox_iou(box1, box2, x1y1x2y2=True):
    """Returns the IoU of two bounding boxes."""
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

def evaluate_coco_map(model, data_loader, coco_gt):
    """Evaluate mAP on COCO validation set"""
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare for COCO evaluation
    coco_dt = []
    image_ids = []
    
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            
            # Forward pass with CUDA flag
            predictions = model(imgs, device == torch.device("cuda"))
            
            # Process predictions
            for i in range(batch_size):
                img_id = targets[i]['image_id'].item()
                image_ids.append(img_id)
                
                if len(predictions[i]) == 0:
                    continue
                
                # Convert predictions to COCO format
                for pred in predictions[i]:
                    x1, y1, x2, y2 = [p.item() for p in pred[:4]]
                    conf = pred[4].item()
                    cls_conf = pred[5].item()
                    cls_pred = int(pred[6].item())  # Convert class prediction to integer
                    
                    coco_dt.append({
                        'image_id': img_id,
                        'category_id': cls_pred,
                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        'score': float(conf * cls_conf)
                    })
    
    if len(coco_dt) == 0:
        return 0.0
    
    # Save predictions to temporary file
    _, tmp_file = tempfile.mkstemp()
    with open(tmp_file, 'w') as f:
        json.dump(coco_dt, f)
    
    # Load predictions in COCO format
    coco_pred = coco_gt.loadRes(tmp_file)
    
    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0]  # Return mAP@[0.5:0.95]

import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
import tempfile
