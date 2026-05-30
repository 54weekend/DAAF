# DAAF Object Detection (Public Review Release)

This repository provides a public review version of a DAAF-based object detection framework implemented in PyTorch.

This release is intended for manuscript-review-stage sharing. It preserves the public training and inference interfaces while removing experiment-specific paths, private notes, and unpublished implementation details. A complete research release can be provided after the review process is completed.

## Features

- YOLO-style object detection pipeline
- Dual-branch feature extraction interface
- Public DAAF fusion interface
- Training, inference, evaluation, and ONNX export scripts
- Generic dataset paths and checkpoint management

## Repository Structure

```text
.
├── train.py
├── predict.py
├── yolo.py
├── nets/
│   ├── yolo.py
│   ├── yolo_training.py
│   ├── backbone.py
│   ├── fusion_modules.py
│   ├── mobilevit.py
│   └── mobilenetv2.py
├── utils/
│   ├── callbacks.py
│   ├── dataloader.py
│   ├── utils.py
│   ├── utils_bbox.py
│   ├── utils_fit.py
│   └── utils_map.py
├── dataset/
│   ├── train.txt
│   └── val.txt
├── model_data/
│   └── classes.txt
├── checkpoints/
├── outputs/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Dataset Format

Prepare annotation files as follows:

```text
path/to/image.jpg x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id
```

Put training and validation annotations in:

```text
dataset/train.txt
dataset/val.txt
```

Put class names in:

```text
model_data/classes.txt
```

## Training

```bash
python train.py
```

Main configuration items are located near the top of `train.py`, including dataset paths, input size, checkpoint path, optimizer, and training schedule.

Checkpoints are saved to:

```text
checkpoints/
```

## Inference

Single image:

```bash
python predict.py --mode image --input assets/demo.jpg --output outputs/result.jpg --weights checkpoints/best.pth --classes model_data/classes.txt
```

Directory:

```bash
python predict.py --mode dir --input assets/images --output outputs/images --weights checkpoints/best.pth --classes model_data/classes.txt
```

Video:

```bash
python predict.py --mode video --input 0 --output outputs/video.avi --weights checkpoints/best.pth --classes model_data/classes.txt
```

## ONNX Export

```bash
python predict.py --mode onnx --weights checkpoints/best.pth --classes model_data/classes.txt --onnx-path outputs/model.onnx
```

## Review-Stage Note

This public version keeps the DAAF module interface available while omitting private implementation notes and experiment-specific details. The detailed research implementation may be released after the manuscript review process is complete.

## License

This public review package is provided for academic evaluation and reproducibility checks. Please update the license according to your final release policy before making the repository fully public.
