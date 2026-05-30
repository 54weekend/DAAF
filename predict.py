import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo import YOLO


def predict_image(model, image_path, output_path=None):
    image = Image.open(image_path)
    result = model.detect_image(image)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
    else:
        result.show()


def predict_directory(model, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = {'.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'}
    for image_path in tqdm([p for p in input_dir.iterdir() if p.suffix.lower() in suffixes]):
        image = Image.open(image_path)
        result = model.detect_image(image)
        result.save(output_dir / image_path.name)


def predict_video(model, video_path, output_path=None, fps=25.0):
    capture = cv2.VideoCapture(0 if video_path == '0' else video_path)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model.detect_image(Image.fromarray(np.uint8(frame_rgb)))
        frame_out = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        cv2.imshow('video', frame_out)
        if writer is not None:
            writer.write(frame_out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Public inference script for DAAF-based object detection.')
    parser.add_argument('--mode', default='image', choices=['image', 'dir', 'video', 'onnx'])
    parser.add_argument('--input', default='assets/demo.jpg')
    parser.add_argument('--output', default='outputs/result.jpg')
    parser.add_argument('--weights', default='checkpoints/best.pth')
    parser.add_argument('--classes', default='model_data/classes.txt')
    parser.add_argument('--onnx-path', default='outputs/model.onnx')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    model = YOLO(model_path=args.weights, classes_path=args.classes, cuda=args.cuda, fusion_type='daaf')
    if args.mode == 'image':
        predict_image(model, args.input, args.output)
    elif args.mode == 'dir':
        predict_directory(model, args.input, args.output)
    elif args.mode == 'video':
        predict_video(model, args.input, args.output)
    elif args.mode == 'onnx':
        model.convert_to_onnx(simplify=True, model_path=args.onnx_path)


if __name__ == '__main__':
    main()
