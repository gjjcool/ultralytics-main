import os.path
import time

import cv2

from ultralytics import YOLO
from multiprocessing import freeze_support
import toolkit.file_operations as fo
from toolkit.predict_pipeline import PredictPipeline


def main():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("runs/server/train14/weights/best.pt")  # load a pretrained model (recommended for training)

    input_dir = r'D:\Resources\Projects\qhx_ocr\input\for_table\text_and_table\dataset_yolo\images\test'
    output_dir = r'D:\Resources\Projects\qhx_ocr\output\chicun_tat'
    # Use the model
    # model.train(data="__coco128.yaml", epochs=10)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set

    for img_path, file_name in fo.find_all_file_paths(input_dir):
        img = cv2.imread(img_path)
        results = model(img)
        res_img = results[0].plot()
        cv2.imwrite(os.path.join(output_dir, file_name), res_img)

if __name__ == "__main__":
    freeze_support()
    main()

# from ultralytics.utils import ASSETS
# from ultralytics.models.yolo.detect import DetectionPredictor
#
# args = dict(model='yolov8n.pt', source=ASSETS)
# predictor = DetectionPredictor(overrides=args)
# predictor.predict_cli()
