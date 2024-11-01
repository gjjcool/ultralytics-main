import os.path

import cv2

from ultralytics import YOLO
from multiprocessing import freeze_support
import toolkit.file_operations as fo


def main():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("runs/server/train14/weights/best.pt")  # load a pretrained model (recommended for training)

    metrics = model.val(data='__chicun_tat.yaml', split='test', batch=2)



if __name__ == "__main__":
    freeze_support()
    main()

# from ultralytics.utils import ASSETS
# from ultralytics.models.yolo.detect import DetectionPredictor
#
# args = dict(model='yolov8n.pt', source=ASSETS)
# predictor = DetectionPredictor(overrides=args)
# predictor.predict_cli()
