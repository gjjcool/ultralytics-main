import json
import os.path
import shutil

import cv2

from ultralytics import YOLO
from multiprocessing import freeze_support
import toolkit.file_operations as fo
from toolkit.predict import Predictor
from toolkit.predict_pipeline import PredictPipeline
from toolkit.table_extractor_draw import TableExtractor


def calc_overlap_area_of_two_rect(a, b):
    if a[0] > a[2]:
        a = [a[2], a[3], a[0], a[1]]
    if b[0] > b[2]:
        b = [b[2], b[3], b[0], b[1]]

    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    if x_overlap <= 0:
        return 0
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    if y_overlap <= 0:
        return 0
    return x_overlap * y_overlap


def filter_chicun(ex_boxes, in_box, box):
    if box[3] <= in_box[0] or box[0] >= in_box[1] or box[1] >= in_box[2] or box[2] <= in_box[3]:
        return False

    thresh_iou = 0.5
    thresh_contain = 0.9

    for ex_b in ex_boxes:
        ex_b_area = abs(ex_b[0] - ex_b[2]) * abs(ex_b[1] - ex_b[3])
        box_area = abs(box[0] - box[2]) * abs(box[1] - box[3])
        inter_area = calc_overlap_area_of_two_rect(ex_b, box)
        iou_rat = inter_area / (ex_b_area + box_area - inter_area)
        if iou_rat >= thresh_iou:
            return False
        contain_rat = inter_area / box_area
        if contain_rat >= thresh_contain:
            return False

    return True


def draw_img(img, boxes):
    h, w = img.shape[:2]
    clr = (0, 0, 255)
    for box in boxes:
        beg = (int(box[0] * w), int(box[1] * h))
        end = (int(box[2] * w), int(box[3] * h))
        cv2.rectangle(img, beg, end, clr, 2)
    return img


def main():
    input_img = r'D:\Resources\Projects\qhx_ocr\input\chicun\1.png'
    input_json = r'D:\Resources\Projects\qhx_ocr\input\chicun\3.json'
    output_dir = r'D:\Resources\Projects\qhx_ocr\output\chicun'
    fo.new_dir(output_dir)

    exclusive_boxes = []
    inner_box = None
    chicun_boxes = []

    # abc
    model_abc = YOLO("runs/server/train13/weights/best.pt")  # load a pretrained model (recommended for training)
    p = Predictor(cv2.imread(input_img))
    p.cut_img((720, 1280))
    p.predict(model_abc)
    p.nms()
    exclusive_boxes += p.get_all_boxes_normalized()

    # table
    model_table = YOLO("runs/server/train5/weights/best.pt")
    pp = PredictPipeline(input_img)
    results = model_table(pp.get_sml_img())
    for xyxy in results[0].boxes.xyxy.tolist():
        xyxy[0] /= pp.target_size[0]
        xyxy[2] /= pp.target_size[0]
        xyxy[1] /= pp.target_size[1]
        xyxy[3] /= pp.target_size[1]
        exclusive_boxes.append(xyxy)

    # chicun_tat
    model_tat = YOLO("runs/server/train14/weights/best.pt")
    results = model_tat(pp.get_sml_img())
    for xyxy in results[0].boxes.xyxy.tolist():
        xyxy[0] /= pp.target_size[0]
        xyxy[2] /= pp.target_size[0]
        xyxy[1] /= pp.target_size[1]
        xyxy[3] /= pp.target_size[1]
        exclusive_boxes.append(xyxy)

    # inner
    clr_img = cv2.imread(input_img)
    gray_img = cv2.cvtColor(clr_img, cv2.COLOR_BGR2GRAY)
    te = TableExtractor(gray_img, clr_img)
    inner_box = te.get_inner_box_normalized()  # trbl

    with open(input_json, 'r') as json_file:
        data = json.load(json_file)
        for inf_box in data['target']:
            quad = inf_box[0]
            x0 = min(quad[0][0], quad[1][0], quad[2][0], quad[3][0])
            y0 = min(quad[0][1], quad[1][1], quad[2][1], quad[3][1])
            x1 = max(quad[0][0], quad[1][0], quad[2][0], quad[3][0])
            y1 = max(quad[0][1], quad[1][1], quad[2][1], quad[3][1])
            box = [x0,y0,x1,y1]
            # inf_box = [inf_box[0][0], inf_box[0][1], inf_box[1][0], inf_box[1][1]]
            if filter_chicun(exclusive_boxes, inner_box, box):
                chicun_boxes.append(box)
        cv2.imwrite(os.path.join(output_dir, 'chicun1.png'), draw_img(cv2.imread(input_img), chicun_boxes))


if __name__ == "__main__":
    freeze_support()
    main()
