import argparse
import copy
import io
import json
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import ImageFont, ImageDraw, Image
import httpx
import asyncio
from typing import Union

from tqdm import tqdm

from UltralyticsYOLO.toolkit.tag_data import TargetTypeTag

import UltralyticsYOLO.toolkit.file_operations as fo
import UltralyticsYOLO.toolkit.comm_toolkit as comm

from ultralytics import YOLO
from UltralyticsYOLO.toolkit.predict import Predictor as Det_ABC
from UltralyticsYOLO.toolkit.predict import ResultFileGen
from UltralyticsYOLO.toolkit.shrink_predict_pipeline import PredictPipeline as Det_Shrink
from UltralyticsYOLO.toolkit.table_extractor_draw import TableExtractor
from UltralyticsYOLO.toolkit.det_main_table_elem_old import DetAndRecMainTableElem
from UltralyticsYOLO.toolkit.det_index_table_elem import DetAndRecIndexTableElem

import PaddleOCR.rec_abc as Rec_ABC
import PaddleOCR.rec_all as Rec_ALL

# YOLO - ABC det
# yolo_det_abc = YOLO("UltralyticsYOLO/runs/server/train13/weights/best.pt")

# YOLO - table(main/index/id/list) det
# yolo_det_table = YOLO("UltralyticsYOLO/runs/server/train5/weights/best.pt")  [obsolete]
yolo_det_table = YOLO("UltralyticsYOLO/runs/server/classify_table_2/weights/best.pt")


# YOLO - main table elem det
# yolo_det_main_table_elem = YOLO("UltralyticsYOLO/runs/server/train16/weights/best.pt")

# YOLO - index table elem det
# yolo_det_index_table_elem = YOLO("UltralyticsYOLO/runs/server/train17/weights/best.pt")

# YOLO - non-size(text and table) det
# yolo_det_tat = YOLO("UltralyticsYOLO/runs/server/train14/weights/best.pt")

# Paddle - ABC rec (only rec)
# padd_rec_abc = Rec_ABC.get_model()

# Paddle - all text rec
# padd_rec_txt = Rec_ALL.get_model()


def small_area_text_rec(img):
    # res, _ = padd_rec_abc([img])
    # print(res)
    # return res[0][0]
    return ''


def process_image(file_contents: bytes):
    # 将字节流转换为OpenCV图像
    nparr = np.frombuffer(file_contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_bytes(image):
    # 将OpenCV图像转换为字节流
    _, img_encoded = cv2.imencode('.png', image)
    return img_encoded.tobytes()


# async def get_result_img(file: UploadFile = UploadFile(...)):
#     file_contents = await file.read()
#     img = process_image(file_contents)
#     p = Det_ABC(img)
#     p.cut_img((720, 1280))
#     p.predict(yolo_det_abc)  # 预测
#     p.nms()
#     return StreamingResponse(io.BytesIO(cv2_to_bytes(p.draw_img())), media_type="image/png")


pageIdSeparator = ','
callbackPatience = 3
callbackInterval = 5


def predict_core(data_dict, img, index, pageId):
    data_dict.append_page(img, pageId)

    # # 检测 ABC 类
    # det_abc = Det_ABC(img)
    # det_abc.cut_img((720, 1280))
    # det_abc.predict(yolo_det_abc)
    # det_abc.nms()
    # # data_dict = det_abc.gen_dict()
    # data_dict.append_boxes(index, det_abc.get_boxes())
    #
    # # 识别 ABC 类
    # Rec_ABC.rec_abc(img, data_dict.get_page(index), padd_rec_abc)

    # 检测 主表 / 明细表
    det_shrink = Det_Shrink(img)
    det_shrink.predict_table(yolo_det_table, data_dict.get_page(index))

    # # 检测并识别 主表元素
    # DetAndRecMainTableElem(img, data_dict.get_page(index), yolo_det_main_table_elem, small_area_text_rec)
    #
    # # 检测并识别 明细表元素
    # DetAndRecIndexTableElem(img, data_dict.get_page(index), yolo_det_index_table_elem, small_area_text_rec)
    #
    # # 检测 非尺寸因素
    # det_shrink.predict_tat(yolo_det_tat)
    #
    # # 界定外边界
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # inner_box = TableExtractor(gray_img, img).get_inner_box_normalized()
    #
    # # 识别 所有文本元素
    # all_text_data = Rec_ALL.rec_all(img, padd_rec_txt)
    #
    # # 筛选出尺寸
    # det_shrink.filter_chicun(all_text_data, data_dict.get_page(index), inner_box)


async def mainTableRec(file: UploadFile):
    data_dict = ResultFileGen('TEST')
    file_contents = await file.read()
    img = process_image(file_contents)
    predict_core(data_dict, img, 0, 'TEST')

    filter_field(data_dict.data)

    return JSONResponse(content=data_dict.data)


def filter_field(data_dict):
    for page in data_dict['pageData']:
        for i in range(len(page['targets'])):
            target = page['targets'][i]
            if 'quad' in target:
                target.pop('quad')
            if 'confidence' in target:
                target.pop('confidence')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()
    fo.clear_or_new_dir(args.output)

    fo.new_dir(os.path.join(args.output, str(TargetTypeTag.mainTable)))
    fo.new_dir(os.path.join(args.output, str(TargetTypeTag.indexTable)))
    fo.new_dir(os.path.join(args.output, str(TargetTypeTag.listTable)))
    fo.new_dir(os.path.join(args.output, str(TargetTypeTag.temp_apartListMainTable)))
    fo.new_dir(os.path.join(args.output, str(TargetTypeTag.temp_apartDeviceMainTable)))

    for img_path, file_name in tqdm(fo.find_all_file_paths(args.input)):
        data_dict = ResultFileGen('TEST')
        img = cv2.imread(img_path)
        rawImg = cv2.imread(img_path)
        predict_core(data_dict, img, 0, 'TEST')
        cntList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for page in data_dict.data['pageData']:
            for t in page['targets']:
                cls = int(t['class'])
                if cls == 3 or cls == 4 or cls == 6 or cls >= 10:
                    cv2.imwrite(os.path.join(
                        os.path.join(args.output, str(cls), f'{fo.get_stem(file_name)}[{cls}_{cntList[cls]}].png')),
                                comm.extractImg(rawImg, t['box']))
