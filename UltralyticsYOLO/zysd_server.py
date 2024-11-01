import copy
import io
import json
import os
import shutil
import tempfile
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

from ultralytics import YOLO
from UltralyticsYOLO.toolkit.predict import Predictor as Det_ABC
from UltralyticsYOLO.toolkit.predict import ResultFileGen
from UltralyticsYOLO.toolkit.shrink_predict_pipeline import PredictPipeline as Det_Shrink
from UltralyticsYOLO.toolkit.table_extractor_draw import TableExtractor
from UltralyticsYOLO.toolkit.det_main_table_elem import DetAndRecMainTableElem
from UltralyticsYOLO.toolkit.det_index_table_elem import DetAndRecIndexTableElem
from UltralyticsYOLO.toolkit.det_list_table_elem import DetAndRecListTableElem
from UltralyticsYOLO.toolkit.det_size_elem import DetAndRecSizeElem
from UltralyticsYOLO.toolkit.det_main_table_elem import MainTableType
from UltralyticsYOLO.toolkit.main_table_html_gen import HTTPMainTableHTMLGen
from UltralyticsYOLO.toolkit.list_table_html_gen import HTTPListTableHTMLGen
from UltralyticsYOLO.toolkit.index_table_html_gen import HTTPIndexTableHTMLGen

import UltralyticsYOLO.toolkit.pdf_2_img as pdf2img
import UltralyticsYOLO.toolkit.draw_box as drawBox
import UltralyticsYOLO.toolkit.file_operations as fo
import PaddleOCR.rec_abc as Rec_ABC
import PaddleOCR.rec_all as Rec_ALL

import UltralyticsYOLO.toolkit.comm_toolkit as comm

with open('zysd_server_conf.json', 'r', encoding='utf-8') as conf_file:
    conf = json.load(conf_file)

app = FastAPI()

# YOLO - ABC det
yolo_det_abc = YOLO(conf["yolo_det_abc"])

# YOLO - table(main/index/id/list) det
yolo_det_table = YOLO(conf["yolo_det_table"])

# YOLO - main table elem det
yolo_det_main_table_elem = YOLO(conf["yolo_det_main_table_elem"])

# YOLO - index table elem det
yolo_det_index_table_elem = YOLO(conf["yolo_det_index_table_elem"])

# YOLO - list table elem det
yolo_det_list_table_elem = YOLO(conf["yolo_det_list_table_elem"])

# YOLO - non-size(text and table) det
yolo_det_tat = YOLO(conf["yolo_det_tat"])

# Paddle - ABC rec (only rec)
padd_rec_abc = Rec_ABC.get_model()

# Paddle - small area rec
padd_rec_sml = Rec_ALL.get_sml_model()

# Paddle - all text rec
padd_rec_txt = Rec_ALL.get_model()

# 不包含尺寸元素的图纸类型（由主表类型界定）
noSizeElemMainTypes = [
    MainTableType.apartListMain,
    MainTableType.apartDeviceMain,
    MainTableType.listMain,
    MainTableType.singleRowMain
]


def small_area_text_rec(img, test=False, isDet=True):
    def det_and_rec(image):
        boxes, res, _ = padd_rec_sml(image)
        if len(res) == 0:
            return ''

        if len(res) == 1:
            return res[0][0]

        resPack = sorted(list(zip(boxes, res)), key=lambda x: x[0][3][0])
        sb = ''
        for _, t in resPack:
            sb += t[0]
        return sb

    def rec(image):
        res, _ = padd_rec_abc([image])
        return res[0][0]

    if not isDet:
        return rec(img)

    text = det_and_rec(img)
    if text.strip() == '':
        return rec(img)
    return text

    # ---
    # if test:
    #     cv2.imwrite(r'D:\Resources\Projects\Python\ultralytics-main-2\val\cell2' + f"\\[{res[0][0]}]{comm.get_time_str()}.png", img)




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


@app.post('/get_result_img')
async def get_result_img(file: UploadFile = UploadFile(...)):
    file_contents = await file.read()
    img, rgb_img = process_image(file_contents)
    p = Det_ABC(img)
    p.cut_img((720, 1280))
    p.predict(yolo_det_abc)  # 预测
    p.nms()
    return StreamingResponse(io.BytesIO(cv2_to_bytes(p.draw_img())), media_type="image/png")


pageIdSeparator = ','
callbackPatience = 5
callbackInterval = 5


def TEST_only_classify_table(data_dict, img, index, pageId, raw_img):
    data_dict.append_page(img, pageId)

    # 检测 主表 / 明细表
    det_shrink = Det_Shrink(img)
    det_shrink.predict_table(yolo_det_table, data_dict.get_page(index))


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
        # # 检测 非尺寸因素
        # det_shrink.predict_tat(yolo_det_tat)

        # 界定外边界
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inner_box = TableExtractor(gray_img, img).get_inner_box_normalized()

        # # 识别 所有文本元素
        # # all_text_data = Rec_ALL.rec_all(img, padd_rec_txt)
        # all_text_data = Rec_ALL.rec_all(raw_img, padd_rec_txt)

        # # 筛选出尺寸
        # det_shrink.filter_chicun(all_text_data, data_dict.get_page(index), inner_box)

        # new
        DetAndRecSizeElem(raw_img, data_dict.get_page(index), padd_rec_txt, yolo_det_tat, inner_box)


async def predict_bt(files, pageIdList, graphId, callbackUrl):
    data_dict = None
    try:
        pageIdList = pageIdList.split(pageIdSeparator)
        data_dict = ResultFileGen(graphId)

        for index, file_contents in enumerate(files):
            # file_contents = await file.read()
            img, rgb_img = process_image(file_contents)
            predict_core(data_dict, img, index, pageIdList[index], rgb_img)

        # 过滤无用字段
        filter_field(data_dict.data)
    except Exception as e:
        data_dict.data['isSuccess'] = False
        data_dict.data['errorInfo'] = traceback.format_exc()
        # callbackCnt = 0
        # async with httpx.AsyncClient() as client:
        #     while callbackCnt < callbackPatience:
        #         callbackCnt += 1
        #         response = await client.post(callbackUrl, json={'errorInfo': str(e)}, timeout=30)
        #         if response.status_code == 200:
        #             print('ERROR_INFO_INFORM_SUCCESS____')
        #             return
        #         print('ERROR_INFO_INFORM____')
        #         await asyncio.sleep(callbackInterval)
        #     return
    finally:
        # 回调
        callbackCnt = 0
        async with httpx.AsyncClient() as client:
            while callbackCnt < callbackPatience:
                callbackCnt += 1
                response = await client.post(callbackUrl, json=data_dict.data, timeout=30)
                print('------RESPONSE--------')
                print("Status code:", response.status_code)
                print("Response JSON:", response.json())
                print("Response text:", response.text)
                print("Response content:", response.content)
                print("Response headers:", response.headers)
                if response.status_code == 200:
                    print('CALLBACK_SUCCESS____')
                    return
                print('CALLBACK____')
                await asyncio.sleep(callbackInterval)


@app.post('/predict')
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


@app.post('/[TEST]mainTableRec')
async def mainTableRec(file: UploadFile):
    data_dict = ResultFileGen('TEST')
    file_contents = await file.read()
    img, rgb_img = process_image(file_contents)
    predict_core(data_dict, img, 0, 'TEST', rgb_img)

    filter_field_test(data_dict.data)

    return JSONResponse(content=data_dict.data)


@app.post('/[TEST]predict')
async def test_predict(files: list[UploadFile]):
    """

    Args:
        files: 支持上传多个文件（支持文件类型：pdf、zip压缩包、png等其他图片格式）

    Returns:
        包含关键标识标注的效果图和相关表格的HTML文件的压缩包（等待推理完毕后点击 Download file）

    """
    # all_draw_img_list = []
    # all_html_list = []

    final_files = []
    for file in files:
        if file.filename.endswith('.zip'):
            file_bytes = io.BytesIO(await file.read())
            with zipfile.ZipFile(file_bytes, 'r') as zip_ref:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # zip_ref.extractall(temp_dir)
                    zip_ref.extractall(temp_dir)
                    # for zip_info in zip_ref.infolist():
                    #     with zip_ref.open(zip_info) as source, open(
                    #             os.path.join(temp_dir, zip_info.filename.encode('cp437').decode('gbk')),
                    #             'wb') as target:
                    #         shutil.copyfileobj(source, target)

                    for file_path, file_name in fo.find_all_file_paths(temp_dir):
                        final_files.append({
                            'filename': file_name.encode('cp437').decode('gbk'),
                            'content': Path(file_path).read_bytes(),
                            'type': fo.get_suffix(file_name)
                        })

                    # for root, _, filenames in os.walk(temp_dir):
                    #     for filename in filenames:
                    #         # filename = filename.encode('cp437').decode('gbk')
                    #         file_path = os.path.join(root, filename)
                    #         final_files.append({
                    #             'filename': filename,
                    #             'content': Path(file_path).read_bytes(),
                    #             'type': fo.get_suffix(filename)
                    #         })
        else:
            content = await file.read()
            final_files.append({
                "filename": file.filename,
                "content": content,
                'type': fo.get_suffix(file.filename)
            })

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        i = 0
        for file in final_files:
            i += 1
            # file_contents = await file.read()
            file_contents = file['content']
            filename = fo.get_stem(file['filename'])
            print(f'--- filename: {filename} ---')
            imgs = []
            rgb_imgs = []
            if file['type'] == '.pdf':
                imgs, rgb_imgs = pdf2img.http_multi_page_pdf_convert_to_img(file_contents)
            else:
                img, rgb_img = process_image(file_contents)
                imgs.append(img)
                rgb_imgs.append(rgb_img)

            j = 0
            for img, rgb_img in zip(imgs, rgb_imgs):
                j += 1
                data_dict = ResultFileGen('TEST')

                # TEST_only_classify_table(data_dict, img, 0, 'TEST', rgb_img)
                predict_core(data_dict, img, 0, 'TEST', rgb_img)

                filter_field(data_dict.data, ['quad', 'confidence'])

                draw_img = drawBox.drawBoxes(rgb_img, data_dict.data)

                # html_list = []
                html_list = [HTTPMainTableHTMLGen(data_dict.get_page(0)).html]
                html_list += HTTPListTableHTMLGen(data_dict.get_page(0)).htmlList
                html_list += HTTPIndexTableHTMLGen(data_dict.get_page(0)).htmlList

                prefix = f'{i}_{j}_'
                zip_file.writestr(f'{prefix}{filename}_image.png', cv2_to_bytes(draw_img))
                k = 0
                for html in html_list:
                    if html is None:
                        continue
                    k += 1
                    zip_file.writestr(f'{prefix}{filename}_table[{k}].html', html)

    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, media_type='application/zip',
                             headers={"Content-Disposition": f"attachment; filename={quote(filename)}.zip"})


def filter_field(data_dict, trivial_field=None):
    if trivial_field is None:
        trivial_field = ['quad', 'confidence', 'mainType', 'isApart', 'tableName']

    for page in data_dict['pageData']:
        for i in range(len(page['targets'])):
            target = page['targets'][i]
            for field in trivial_field:
                if field in target:
                    target.pop(field)
            # if 'quad' in target:
            #     target.pop('quad')
            # if 'confidence' in target:
            #     target.pop('confidence')
            # if 'mainType' in target:
            #     target.pop('mainType')

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


def filter_field_test(data_dict):
    for page in data_dict['pageData']:
        for i in range(len(page['targets'])):
            target = page['targets'][i]
            if 'quad' in target:
                target.pop('quad')
            # if 'confidence' in target:
            #     target.pop('confidence')

            # 去首尾空字符
            if 'text' in target:
                text = target['text']
                if isinstance(text, str):
                    target['text'] = text.strip()
                elif isinstance(text, list):
                    for i in range(len(text)):
                        for k, v in target['text'][i].items():
                            if isinstance(v, str):
                                target['text'][i][k] = v.strip()
                elif isinstance(text, dict):
                    for k, v in target['text'].items():
                        if isinstance(v, str):
                            target['text'][k] = v.strip()


def local_test(img_path):
    img = cv2.imread(img_path)
    data_dict = ResultFileGen('TEST')
    predict_core(data_dict, img, 0, 'TEST', img)


def main():
    import uvicorn

    uvicorn.run('zysd_server:app', host=conf['host'], port=conf['port'], reload=True)
    # uvicorn.run(app, host=conf['host'], port=conf['port'])
    # uvicorn.run(app, host='10.108.22.35', port=8000)


if __name__ == "__main__":
    main()
    # local_test(r'D:\Resources\Projects\drawing_rec\input\2_batch\mingxibiao_png\82840138ID3007ME011-3-1-A.png')
