import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def bbox_ciou(box1, box2, x1y1x2y2=True):
    """
    Calculate CIoU loss between two bounding boxes
    """
    if not x1y1x2y2:
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + 1e-16

    iou = inter / union

    # Get enclosed coordinates
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    # Get diagonal distance
    c2 = cw ** 2 + ch ** 2 + 1e-16

    # Get center distance
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + 
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    # Calculate v and alpha
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-16)) - torch.atan(w1 / (h1 + 1e-16)), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + 1e-16))

    # CIoU
    return iou - (rho2 / c2 + v * alpha)

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

class YOLOv4Loss(nn.Module):
    def __init__(self, anchors, num_classes=80):
        super(YOLOv4Loss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: tensor of shape (batch_size, num_boxes, 5 + num_classes)
            targets: list of dicts containing boxes and labels
        """
        device = predictions.device
        batch_size = predictions.size(0)
        total_loss = torch.tensor(0., requires_grad=True, device=device)
        
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            
            # Get target boxes and labels and move them to the correct device
            target_boxes = target['boxes'].to(device)
            target_labels = target['labels'].to(device)
            
            if len(target_boxes) == 0:
                continue
            
            # Get prediction components
            pred_boxes = pred[..., :4]  # [x, y, w, h]
            pred_conf = pred[..., 4]    # objectness
            pred_cls = pred[..., 5:]    # class scores
            
            # Calculate IoU for each predicted box with each target box
            num_pred = pred_boxes.size(0)
            num_target = target_boxes.size(0)
            
            # Expand dimensions for broadcasting
            pred_boxes = pred_boxes.unsqueeze(1).repeat(1, num_target, 1)
            target_boxes = target_boxes.unsqueeze(0).repeat(num_pred, 1, 1)
            
            # Calculate CIoU loss
            ciou = bbox_ciou(pred_boxes.view(-1, 4), target_boxes.view(-1, 4))
            ciou = ciou.view(num_pred, num_target)
            
            # For each target, find the best matching prediction
            best_ious, best_idx = ciou.max(dim=0)
            
            # Calculate box loss using CIoU
            box_loss = (1.0 - best_ious).mean()
            
            # Calculate objectness loss
            obj_mask = torch.zeros_like(pred_conf)
            obj_mask[best_idx] = 1
            obj_loss = F.binary_cross_entropy_with_logits(pred_conf, obj_mask)
            
            # Calculate classification loss
            target_cls = torch.zeros_like(pred_cls)
            for j, label in enumerate(target_labels):
                target_cls[best_idx[j], label] = 1
            cls_loss = F.binary_cross_entropy_with_logits(pred_cls, target_cls)
            
            # Combine losses
            batch_loss = box_loss + obj_loss + cls_loss
            total_loss = total_loss + batch_loss
        
        return total_loss / batch_size
    
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)
