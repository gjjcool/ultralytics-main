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
    model = YOLO("runs/server/train5/weights/best.pt")  # load a pretrained model (recommended for training)

    input_dir = r'D:\Resources\Projects\qhx_ocr\input\录屏240112\table_input'
    output_dir = r'D:\Resources\Projects\qhx_ocr\input\录屏240112\table_output'
    txt_path = os.path.join(output_dir, 'duration.txt')
    table_dir = os.path.join(output_dir, 'table')
    res_dir = os.path.join(output_dir, 'res')
    fo.clear_or_new_dir(table_dir)
    fo.clear_or_new_dir(res_dir)
    # Use the model
    # model.train(data="__coco128.yaml", epochs=10)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    with open(txt_path, 'w') as txt_file:
        cnt = 0
        sum = 0
        for img_path, file_name in fo.find_all_file_paths(input_dir):
            cnt += 1
            pp = PredictPipeline(img_path)

            st = time.time()
            results = model(pp.get_sml_img())
            dur = time.time() - st
            sum += dur
            txt_file.write(f'{file_name}\t{dur}')

            main_imgs, index_imgs = pp.get_tables_on_raw_img(results[0].boxes.xyxy.tolist(), results[0].boxes.cls.tolist())
            res_img = results[0].plot()
            cv2.imwrite(os.path.join(res_dir, file_name), res_img)
            for i in range(len(main_imgs)):
                cv2.imwrite(os.path.join(table_dir, f'{fo.get_stem(file_name)}_main_{i}.png'), main_imgs[i])
            for i in range(len(index_imgs)):
                cv2.imwrite(os.path.join(table_dir, f'{fo.get_stem(file_name)}_index_{i}.png'), index_imgs[i])
        txt_file.write(f'average: {sum / cnt}')

if __name__ == "__main__":
    freeze_support()
    main()

# from ultralytics.utils import ASSETS
# from ultralytics.models.yolo.detect import DetectionPredictor
#
# args = dict(model='yolov8n.pt', source=ASSETS)
# predictor = DetectionPredictor(overrides=args)
# predictor.predict_cli()
