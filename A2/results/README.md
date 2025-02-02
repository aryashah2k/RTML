# Image Inference using YOLOv4 weights will be stored here 

Script commnand to run : 
```python
python main.py --img data/mini_coco/mini_train2017/000000001146.jpg --weights weights/yolov4.weights
```

Expected console output:
```bash
jupyter-st125462@puffer:~/RTML/A2/YOLOv4_Project$ python main.py --img data/mini_coco/mini_train2017/000000001146.jpg --weights weights/yolov4.weights
Loading model...
Loading weights from weights/yolov4.weights
Loading image: /home/jupyter-st125462/RTML/A2/YOLOv4_Project/data/mini_coco/mini_train2017/000000001146.jpg
Found 2 objects!
Class: person (0), Confidence: 1.0000, Box: [64.9, -74.2, 525.9, 550.7]
Class: tie (27), Confidence: 0.9999, Box: [380.0, 52.7, 479.5, 410.4]

Detection result saved to: /home/jupyter-st125462/RTML/A2/YOLOv4_Project/results/000000001146_detection.jpg
```