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
yolo_det_abc = YOLO("UltralyticsYOLO/runs/server/train13/weights/best.pt")

# YOLO - table(main/index/id/list) det
# yolo_det_table = YOLO("UltralyticsYOLO/runs/server/train5/weights/best.pt")  [obsolete]
yolo_det_table = YOLO("UltralyticsYOLO/runs/server/train18/weights/best.pt")

# YOLO - main table elem det
yolo_det_main_table_elem = YOLO("UltralyticsYOLO/runs/server/train16/weights/best.pt")

# YOLO - index table elem det
yolo_det_index_table_elem = YOLO("UltralyticsYOLO/runs/server/train17/weights/best.pt")

# YOLO - non-size(text and table) det
yolo_det_tat = YOLO("UltralyticsYOLO/runs/server/train14/weights/best.pt")

# Paddle - ABC rec (only rec)
padd_rec_abc = Rec_ABC.get_model()

# Paddle - all text rec
padd_rec_txt = Rec_ALL.get_model()


def small_area_text_rec(img):
    res, _ = padd_rec_abc([img])
    print(res)
    return res[0][0]


def process_image(file_contents: bytes):
    # 将字节流转换为OpenCV图像
    nparr = np.frombuffer(file_contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_bytes(image):
    # 将OpenCV图像转换为字节流
    _, img_encoded = cv2.imencode('.png', image)
    return img_encoded.tobytes()


async def get_result_img(file: UploadFile = UploadFile(...)):
    file_contents = await file.read()
    img = process_image(file_contents)
    p = Det_ABC(img)
    p.cut_img((720, 1280))
    p.predict(yolo_det_abc)  # 预测
    p.nms()
    return StreamingResponse(io.BytesIO(cv2_to_bytes(p.draw_img())), media_type="image/png")


pageIdSeparator = ','
callbackPatience = 3
callbackInterval = 5


def predict_core(data_dict, img, index, pageId):
    data_dict.append_page(img, pageId)

    # 检测 ABC 类
    det_abc = Det_ABC(img)
    det_abc.cut_img((720, 1280))
    det_abc.predict(yolo_det_abc)
    det_abc.nms()
    # data_dict = det_abc.gen_dict()
    data_dict.append_boxes(index, det_abc.get_boxes())

    # 识别 ABC 类
    Rec_ABC.rec_abc(img, data_dict.get_page(index), padd_rec_abc)

    # 检测 主表 / 明细表
    det_shrink = Det_Shrink(img)
    det_shrink.predict_table(yolo_det_table, data_dict.get_page(index))

    # 检测并识别 主表元素
    DetAndRecMainTableElem(img, data_dict.get_page(index), yolo_det_main_table_elem, small_area_text_rec)

    # 检测并识别 明细表元素
    DetAndRecIndexTableElem(img, data_dict.get_page(index), yolo_det_index_table_elem, small_area_text_rec)

    # 检测 非尺寸因素
    det_shrink.predict_tat(yolo_det_tat)

    # 界定外边界
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inner_box = TableExtractor(gray_img, img).get_inner_box_normalized()

    # 识别 所有文本元素
    all_text_data = Rec_ALL.rec_all(img, padd_rec_txt)

    # 筛选出尺寸
    det_shrink.filter_chicun(all_text_data, data_dict.get_page(index), inner_box)


async def predict_bt(files, pageIdList, graphId, callbackUrl):
    pageIdList = pageIdList.split(pageIdSeparator)
    data_dict = ResultFileGen(graphId)

    for index, file_contents in enumerate(files):
        # file_contents = await file.read()
        img = process_image(file_contents)
        predict_core(data_dict, img, index, pageIdList[index])

    # 过滤无用字段
    filter_field(data_dict.data)

    # 回调
    callbackCnt = 0
    async with httpx.AsyncClient() as client:
        while callbackCnt < callbackPatience:
            callbackCnt += 1
            response = await client.post(callbackUrl, json=data_dict.data, timeout=30)
            if response.status_code == 200:
                print('CALLBACK_SUCCESS____')
                return
            print('CALLBACK____')
            await asyncio.sleep(callbackInterval)

    # return JSONResponse(content=data_dict.data)


async def predict(background_tasks: BackgroundTasks,
                  files: list[UploadFile],
                  pageIdList: str = Form(),
                  graphId: str = Form(),
                  callbackUrl: str = Form()):
    """
    预测多张图纸页方法
    Args:
        files: 多张图纸列表
        pageIdList: 多张图纸页编号拼接字符串（用英文逗号[,]分隔）
        graphId: 本次预测图纸请求编号
        callbackUrl: 回调 URL，用于接收图纸识别信息的客户端接口

    Returns: True（若正确接受到客户端请求）

    """
    if callbackUrl is None:
        callbackUrl = r'http://dev.drawing.cisdi.sczlcq.com/api/bizGraph/analysisCallback'
    print(callbackUrl)

    for file in files:
        content_type = file.content_type
        filename = file.filename
        print(f'file type: {content_type}, name: {filename}')

    files_data = []
    for file in files:
        files_data.append(await file.read())

    # background_tasks.add_task(predict_bt, copy.deepcopy(files), pageIdList, graphId, callbackUrl)
    background_tasks.add_task(predict_bt, files_data, pageIdList, graphId, callbackUrl)

    print('RETURN____')
    return True


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


palette = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 152, 255),
]


def drawBoxes(img, data_dict):
    text = []
    for t in data_dict['pageData'][0]['targets']:
        cls = int(t['class'])
        absBox = comm.getAbsBoxByImg(t['box'], img)
        if cls <= 2 or cls == 5:
            img = cv2.rectangle(img, absBox[0], absBox[1], palette[cls], 2)
            # cv2.putText(img, t['text'], absBox[1], font, 1.0, (0, 0, 255), 2)
            text.append(((absBox[0][0], absBox[1][1]), t['text']))
        elif cls == 3:
            img = cv2.rectangle(img, absBox[0], absBox[1], palette[cls], 4)
            for k, v in t['cellBoxes'].items():
                absBox = comm.getAbsBoxByImg(v, img)
                img = cv2.rectangle(img, absBox[0], absBox[1], (152, 0, 255), 2)
        elif cls == 4:
            img = cv2.rectangle(img, absBox[0], absBox[1], palette[cls], 4)
            for boxes in t['cellBoxes']:
                for k, v in boxes.items():
                    absBox = comm.getAbsBoxByImg(v, img)
                    img = cv2.rectangle(img, absBox[0], absBox[1], (152, 255, 0), 2)

    img2 = Image.fromarray(img)
    drawImg = ImageDraw.Draw(img2)
    for pos, t in text:
        drawImg.text(pos, t, font=font, fill=(0, 0, 255))
    return np.array(img2)


if __name__ == "__main__":
    font = ImageFont.truetype('simfang.ttf', 40)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_path = 'simfang.ttf'
    # cv2.fontQt.addFont(font_path, font)

    input_dir = r'D:\Resources\Projects\drawing_rec\output\test20240508\789'
    output_dir = r'D:\Resources\Projects\drawing_rec\output\test20240508\res789'
    fo.clear_or_new_dir(output_dir)

    for img_path, file_name in fo.find_all_file_paths(input_dir):
        data_dict = ResultFileGen('TEST')
        img = cv2.imread(img_path)
        predict_core(data_dict, img, 0, 'TEST')
        img = drawBoxes(img, data_dict.data)
        cv2.imwrite(os.path.join(output_dir, file_name), img)
        with open(os.path.join(output_dir, fo.get_stem(file_name) + '.json'), 'w') as json_file:
            json.dump(data_dict.data, json_file)
