# ILENet
ILENet: Illumination-Modulated Laplacian-Pyramid Enhancement Network for Low-Light Object Detection

By combining our proposed low-light enhancement network ILENet with the YOLO detector, we form a new, powerful single-stage detection model. Using a joint training strategy, we effectively balance the low-light image enhancement task and object detection task, thereby significantly improving detection performance.

## The operating results of this project refer to:  
Quantitative comparisons of ILE-YOLO and other LLIE methods, as well as the latest detection models, on the ExDark dataset, with the best and second-best results marked in red and blue, respectively

| Algorithm     | Bicycle   | Boat   | Bottle   | Bus   | Car   | Cat      | Chair      | Cup      | Dog   | Motorbike  | People  | Table   | mAP      |
|---------------|-----------|--------|----------|-------|-------|----------|------------|----------|-------|------------|---------|---------|----------| 
| YOLOv3        | 79.8 |75.3 |78.1 |92.3 |83.0| 68.0 |69.0 |79.0 |78.0 |77.3 |81.5 |55.5 | 76.4     |
| KIND| 80.1 | 77.7 | 77.2 | 93.8 | 83.9 | 66.9 | 68.7 | 77.4 | 79.3 | 75.3 | 80.9 | 53.8 | 76.3     | 
| MBLLEN| 82.0 | 77.3 | 76.5 | 91.3 | 84.0 | 67.6 | 69.1 | 77.6 | 80.4 | 75.6 | 81.9 | 58.6 | 76.8     |
| Zero-DCE | 84.1 | 77.6 | 78.3 | 93.1 | 83.7 | 70.3 | 69.8 | 77.6 | 77.4 | 76.3 | 81.0 | 53.6 | 76.9     |
| YOLOv10  | 81.0 | 78.6 | 76.1 | 91.6 | 83.8 | 72.2 | 72.4 | 78.4 | 76.8 | 76.7 | 81.6 | 56.6 | 77.1     |
| YOLOv11 | 80.5 | 77.8 | 77.4 | 93.0 | 85.1 | 71.9 | 72.3 | 80.0 | 79.1 | 77.1 | 81.5 | 54.6 | 77.5     |
| Retinexformer  | 84.3 | 77.6 | 77.7 | 92.4 | 83.4 | 71.2 | 73.1 | 79.2 | 79.2 | 77.0 | 83.0 | 56.1 | 77.9     |
| **ILE-YOLO (Ours)** | **83.7** | **78.2** | **78.2** | **92.6** | **84.1** | **71.3** | **74.3** | **80.1** | **80.0** | **77.4** | **83.2** | **58.3** | **78.5** |

Low-Light Object Detection

| Algorithm         | Bicycle    | Boat   | Bottle   | Bus   | Car   | Cat      | Chair      | Cup      | Dog   | Motorbike  | People  | Table   | mAP   | Times     |
|-------------------|------------|--------|----------|-------|-------|----------|------------|----------|-------|------------|---------|---------|-------|-----------|
| YOLOv3            | 79.8 |75.3 |78.1 |92.3 |83.0| 68.0 |69.0 |79.0 |78.0 |77.3 |81.5 |55.5 | 76.4     | 0.033     |
| DENet             | 80.4 | 79.7 | 77.9 | 91.2 | 82.7 | 72.8 | 69.9 | 80.1 | 77.2 | 76.7 | 82.0 | 57.2 | 77.3 | 0.037     |
| PE-YOLO           | 83.2 | 77.8 | 78.6 | 93.0 | 83.7 | 71.2 | 71.0 | 79.6 | 79.2 | 77.0 | 82.0 | 55.5 | 77.6 | 0.125     | 
| MAET              | 83.1 | 78.5 | 75.6 | 92.9 | 83.1 | 73.4 | 71.3 | 79.0 | 79.8 | 77.2 | 81.1 | 57.0 | 77.7 | -         | 
| IAT-YOLO          | 79.8 | 76.9 | 78.6 | 92.5 | 83.8 | 73.6 | 72.4 | 78.6 | 79.0 | 79.0 | 81.1 | 57.7 | 77.8 | 0.040     |
| **ILE-YOLO (Ours)** | **83.7** | **78.2** | **78.2** | **92.6** | **84.1** | **71.3** | **74.3** | **80.1** | **80.0** | **77.4** | **83.2** | **58.3** | **78.5** | **0.047** |


## Dependencies
```
pip install -r requirements.txt
````

## Install BasicSR
```
pip install timm
```

## Run command:    
```
python tools/train.py
```
