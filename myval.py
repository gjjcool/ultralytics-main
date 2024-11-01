import copy
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.requests import Request
import httpx
import asyncio
import zipfile
from urllib.parse import quote
import traceback
from typing import Union

from tqdm import tqdm
from ultralytics import YOLO
from UltralyticsYOLO.toolkit.predict import Predictor as Det_ABC
from UltralyticsYOLO.toolkit.predict import ResultFileGen
from UltralyticsYOLO.toolkit.shrink_predict_pipeline import PredictPipeline as Det_Shrink
from UltralyticsYOLO.toolkit.table_extractor_draw import TableExtractor
from UltralyticsYOLO.toolkit.det_main_table_elem import DetAndRecMainTableElem
from UltralyticsYOLO.toolkit.det_index_table_elem import DetAndRecIndexTableElem
from UltralyticsYOLO.toolkit.det_list_table_elem import DetAndRecListTableElem
from UltralyticsYOLO.toolkit.det_main_table_elem import MainTableType
from UltralyticsYOLO.toolkit.main_table_html_gen import HTTPMainTableHTMLGen
from UltralyticsYOLO.toolkit.list_table_html_gen import HTTPListTableHTMLGen
from UltralyticsYOLO.toolkit.index_table_html_gen import HTTPIndexTableHTMLGen
from UltralyticsYOLO.toolkit.tag_data import TargetTypeTag

import UltralyticsYOLO.toolkit.pdf_2_img as pdf2img
import UltralyticsYOLO.toolkit.draw_box as drawBox
import UltralyticsYOLO.toolkit.file_operations as fo
import PaddleOCR.rec_abc as Rec_ABC
import PaddleOCR.rec_all as Rec_ALL

import UltralyticsYOLO.toolkit.comm_toolkit as comm

import UltralyticsYOLO.toolkit.val_toolkit as valtk

app = FastAPI()

# YOLO - ABC det
yolo_det_abc = YOLO("UltralyticsYOLO/runs/server/train13/weights/best.pt")

# YOLO - table(main/index/id/list) det
# yolo_det_table = YOLO("UltralyticsYOLO/runs/server/train5/weights/best.pt")  [obsolete]
yolo_det_table = YOLO("UltralyticsYOLO/runs/server/classify_table_2/weights/best.pt")

# YOLO - main table elem det
yolo_det_main_table_elem = YOLO("UltralyticsYOLO/runs/server/main_elem_2/weights/best.pt")

# YOLO - index table elem det
yolo_det_index_table_elem = YOLO("UltralyticsYOLO/runs/server/index_elem_1/weights/best.pt")

# YOLO - list table elem det
yolo_det_list_table_elem = YOLO("UltralyticsYOLO/runs/server/list_elem_3/weights/best.pt")

# YOLO - non-size(text and table) det
yolo_det_tat = YOLO("UltralyticsYOLO/runs/server/train14/weights/best.pt")

# Paddle - ABC rec (only rec)
padd_rec_abc = Rec_ABC.get_model()

# Paddle - all text rec
padd_rec_txt = Rec_ALL.get_model()

# 不包含尺寸元素的图纸类型（由主表类型界定）
noSizeElemMainTypes = [
    MainTableType.apartListMain,
    MainTableType.apartDeviceMain,
    MainTableType.listMain,
    MainTableType.singleRowMain
]


def small_area_text_rec(img, test=False):
    res, _ = padd_rec_abc([img])
    if test:
        cv2.imwrite(r'D:\Resources\Projects\Python\ultralytics-main-2\val\cell' + f"\\[{res[0][0]}]{comm.get_time_str()}.png", img)
    print(res)
    return res[0][0]


def process_image(file_contents: bytes):
    # 将字节流转换为OpenCV图像
    nparr = np.frombuffer(file_contents, np.uint8)
    rgb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    thresh = 250
    _, binary_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)

    binary_img_3c = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)  # 灰度转 3 通道图片

    return binary_img_3c, rgb_img


def cv2_to_bytes(image):
    # 将OpenCV图像转换为字节流
    _, img_encoded = cv2.imencode('.png', image)
    return img_encoded.tobytes()

def predict_core(data_dict, img, index, pageId, raw_img):
    data_dict.append_page(img, pageId)

    # 检测 ABC 类
    det_abc = Det_ABC(img)
    det_abc.cut_img((720, 1280))
    det_abc.predict(yolo_det_abc)
    det_abc.nms()
    # data_dict = det_abc.gen_dict()
    data_dict.append_boxes(index, det_abc.get_boxes())

    # 识别 ABC 类
    # Rec_ABC.rec_abc(img, data_dict.get_page(index), padd_rec_abc)
    Rec_ABC.rec_abc(raw_img, data_dict.get_page(index), padd_rec_abc)

    # 检测 主表 / 明细表
    det_shrink = Det_Shrink(img)
    det_shrink.predict_table(yolo_det_table, data_dict.get_page(index))

    # 检测并识别 主表元素
    DnR_mainTableElem = DetAndRecMainTableElem(img, data_dict.get_page(index), yolo_det_main_table_elem,
                                               small_area_text_rec, raw_img)

    # 检测并识别 明细表元素
    DetAndRecIndexTableElem(img, data_dict.get_page(index), yolo_det_index_table_elem, small_area_text_rec, raw_img)

    # 检测并识别 目录材料表元素
    DetAndRecListTableElem(img, data_dict.get_page(index), yolo_det_list_table_elem, small_area_text_rec, raw_img)

    # 检测并识别 尺寸元素
    if DnR_mainTableElem.getMainType() not in noSizeElemMainTypes:
        # 检测 非尺寸因素
        det_shrink.predict_tat(yolo_det_tat)

        # 界定外边界
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inner_box = TableExtractor(gray_img, img).get_inner_box_normalized()

        # 识别 所有文本元素
        # all_text_data = Rec_ALL.rec_all(img, padd_rec_txt)
        all_text_data = Rec_ALL.rec_all(raw_img, padd_rec_txt)

        # 筛选出尺寸
        det_shrink.filter_chicun(all_text_data, data_dict.get_page(index), inner_box)


def filter_field(data_dict, trivial_field=None):
    if trivial_field is None:
        trivial_field = ['quad', 'confidence', 'mainType', 'isApart', 'tableName']

    for page in data_dict['pageData']:
        for i in range(len(page['targets'])):
            target = page['targets'][i]
            for field in trivial_field:
                if field in target:
                    target.pop(field)

            # 去首尾空字符
            if 'text' in target:
                text = target['text']
                if isinstance(text, str):
                    target['text'] = text.strip()
                elif isinstance(text, list):  # 明细表/目录表
                    for i in range(len(text)):
                        for k, v in target['text'][i].items():
                            if isinstance(v, str):
                                target['text'][i][k] = v.strip()
                elif isinstance(text, dict):  # 主表
                    for k, v in target['text'].items():
                        if isinstance(v, str):
                            target['text'][k] = v.strip()

            # 去“明细表/目录表”中空行数据项
            if 'text' in target:
                text = target['text']
                if isinstance(text, list):  # 明细表/目录表
                    for i in range(len(text) - 1, -1, -1):
                        isEmptyRow = True
                        for k, v in target['text'][i].items():
                            if isinstance(v, str):
                                if len(v) > 0:
                                    isEmptyRow = False
                                    break
                        if isEmptyRow:
                            del target['text'][i]
                            if 'cellBoxes' in target:
                                del target['cellBoxes'][i]


if __name__ == "__main__":
    timestamp = int(time.time())

    with open('myval_conf.json', 'r', encoding='utf-8') as conf_file:
        conf = json.load(conf_file)
    input_dir = conf['input_dir']
    output_dir = conf['output_dir']

    output_data_dir = os.path.join(output_dir, 'data')
    timestamp_dir = os.path.join(output_data_dir, str(timestamp))
    os.mkdir(timestamp_dir)

    res_data = dict()
    text_res_data = {'TP': 0, 'FP': 0, 'FN': 0}

    truth_labels = dict()
    pred_labels = dict()

    for file_path, file_name in tqdm(fo.find_all_file_paths(input_dir)):
        if fo.get_suffix(file_name) == '.json':
            truth_labels[fo.get_stem(file_name)] = valtk.read_and_process_labelme_json(file_path)

        else:
            data_dict = ResultFileGen('TEST')
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            _, binImg = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
            bin3cImg = cv2.cvtColor(binImg, cv2.COLOR_GRAY2BGR)
            rawImg = cv2.imread(file_path)
            predict_core(data_dict, bin3cImg, 0, 'TEST', rawImg)
            filter_field(data_dict.data, ['quad', 'mainType', 'isApart', 'tableName'])

            pred_labels[fo.get_stem(file_name)] = data_dict.data

    for k in truth_labels.keys():
        with open(os.path.join(timestamp_dir, k + '_truth.json'), 'w', encoding='utf-8') as data_file:
            json.dump(truth_labels[k], data_file, ensure_ascii=False, indent=4)

        box_res, text_res = valtk.validate(truth_labels[k], pred_labels[k])
        valtk.aggregate_box(res_data, box_res)
        valtk.aggregate_text(text_res_data, text_res)

        with open(os.path.join(timestamp_dir, k + '.json'), 'w', encoding='utf-8') as data_file:
            json.dump(pred_labels[k], data_file, ensure_ascii=False, indent=4)

    res_list = [valtk.translate_output_res('字符识别', text_res_data)]
    for k, v in res_data.items():
        res_list.append(valtk.translate_output_res(TargetTypeTag.nameList[k], v))

    json_data = {'res': res_list}

    with open(os.path.join(output_dir, f'{timestamp}.json'), 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)


