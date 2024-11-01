import os.path
import shutil

import cv2

from ultralytics import YOLO
from multiprocessing import freeze_support
import toolkit.file_operations as fo
from toolkit.predict import Predictor


def main():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("runs/server/train13/weights/best.pt")  # load a pretrained model (recommended for training)

    input_dir = r'D:\Resources\Projects\qhx_ocr\input\temp240129\abc_input'
    output_dir = r'D:\Resources\Projects\qhx_ocr\input\temp240129\abc_output'
    img_dir = os.path.join(output_dir, 'images')
    json_dir = os.path.join(output_dir, 'jsons')
    fo.new_dir(output_dir)
    fo.new_dir(img_dir)
    fo.new_dir(json_dir)

    # Use the model
    # model.train(data="__coco128.yaml", epochs=10)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    for img_path, file_name in fo.find_all_file_paths(input_dir):
        p = Predictor(cv2.imread(img_path))
        p.cut_img((720, 1280))
        p.predict(model)
        p.nms()
        cv2.imwrite(os.path.join(img_dir, file_name), p.draw_img())
        p.gen_json(file_name, json_dir)


if __name__ == "__main__":
    freeze_support()
    main()

# from ultralytics.utils import ASSETS
# from ultralytics.models.yolo.detect import DetectionPredictor
#
# args = dict(model='yolov8n.pt', source=ASSETS)
# predictor = DetectionPredictor(overrides=args)
# predictor.predict_cli()
