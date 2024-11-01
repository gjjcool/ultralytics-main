import os.path

import cv2

from ultralytics import YOLO
from multiprocessing import freeze_support
import toolkit.file_operations as fo


def main():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("best.pt")  # load a pretrained model (recommended for training)

    input_dir = r'D:\Resources\Projects\qhx_ocr\input\labelled_B\no_b\no_b_dataset\test'
    output_dir = r'D:\Resources\Projects\qhx_ocr\output\b_enter_2\test_no_b'
    no_dir = r'D:\Resources\Projects\qhx_ocr\output\b_enter_2\test_no_b\no'

    # Use the model
    # model.train(data="__coco128.yaml", epochs=10)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    for img_path, file_name in fo.find_all_file_paths(input_dir):
        results = model(img_path)
        # main_imgs, index_imgs = pp.get_tables_on_raw_img(results[0].boxes.xyxy.tolist(), results[0].boxes.cls.tolist())
        box_cnt = len(results[0].boxes.xyxy.tolist())
        res_img = results[0].plot()
        if box_cnt > 0:
            cv2.imwrite(os.path.join(output_dir, file_name), res_img)
        else:
            cv2.imwrite(os.path.join(no_dir, file_name), res_img)


if __name__ == "__main__":
    freeze_support()
    main()

# from ultralytics.utils import ASSETS
# from ultralytics.models.yolo.detect import DetectionPredictor
#
# args = dict(model='yolov8n.pt', source=ASSETS)
# predictor = DetectionPredictor(overrides=args)
# predictor.predict_cli()
