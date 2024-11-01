import io
import json
import tempfile
import traceback
from datetime import datetime
# from multiprocessing import Process
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
# import numpy as np
# from celery.signals import setup_logging
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.openapi.docs import get_swagger_ui_html
# from minio import Minio
from starlette.requests import Request
# import httpx
# import httpcore
# import asyncio
import zipfile
from urllib.parse import quote
# import traceback
# from typing import Union
# # from concurrent.futures import ThreadPoolExecutor
# from celery import Celery

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
import os
from celery_apps.predict_algo import predict_test_page

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

with open('zysd_server_conf.json', 'r', encoding='utf-8') as conf_file:
    conf = json.load(conf_file)


app = FastAPI()
# cpu_count = os.cpu_count()
# print(f'cpu count: {cpu_count}')
# executor = ThreadPoolExecutor(max_workers=conf['max_workers'])
#
# # YOLO - ABC det
# yolo_det_abc = (YOLO(conf["yolo_det_abc"])
#                 .to(f'cuda:{conf["yolo_det_abc_device"]}'))
# # yolo_det_abc_2 = (YOLO(conf["yolo_det_abc"])
# #                   .to(f'cuda:{conf["yolo_det_abc_2_device"]}'))
#
#
# def yolo_det_abc_pred(img, device=conf["yolo_det_abc_device"]):
#     # return (yolo_det_abc if index % 2 == 0 else yolo_det_abc_2).predict(img)
#     return yolo_det_abc.predict(img, device=f'cuda:{device}')
#
#
# # YOLO - table(main/index/id/list) det
# yolo_det_table = (YOLO(conf["yolo_det_table"])
#                   .to(f'cuda:{conf["yolo_det_table_device"]}'))
#
#
# def yolo_det_table_pred(img, device=conf["yolo_det_table_device"]):
#     return yolo_det_table.predict(img, device=f'cuda:{device}')
#
#
# # YOLO - main table elem det
# yolo_det_main_table_elem = (YOLO(conf["yolo_det_main_table_elem"])
#                             .to(f'cuda:{conf["yolo_det_main_table_elem_device"]}'))
#
#
# def yolo_det_main_table_elem_pred(img, device=conf["yolo_det_main_table_elem_device"]):
#     return yolo_det_main_table_elem.predict(img, device=f'cuda:{device}')
#
#
# # YOLO - index table elem det
# yolo_det_index_table_elem = (YOLO(conf["yolo_det_index_table_elem"])
#                              .to(f'cuda:{conf["yolo_det_index_table_elem_device"]}'))
#
#
# def yolo_det_index_table_elem_pred(img, device=conf["yolo_det_index_table_elem_device"]):
#     return yolo_det_index_table_elem.predict(img, device=f'cuda:{device}')
#
#
# # YOLO - list table elem det
# yolo_det_list_table_elem = (YOLO(conf["yolo_det_list_table_elem"])
#                             .to(f'cuda:{conf["yolo_det_list_table_elem_device"]}'))
#
#
# def yolo_det_list_table_elem_pred(img, device=conf["yolo_det_list_table_elem_device"]):
#     return yolo_det_list_table_elem.predict(img, device=f'cuda:{device}')
#
#
# # YOLO - non-size(text and table) det
# yolo_det_tat = (YOLO(conf["yolo_det_tat"])
#                 .to(f'cuda:{conf["yolo_det_tat_device"]}'))
#
#
# def yolo_det_tat_pred(img, device=conf["yolo_det_tat_device"]):
#     return yolo_det_tat.predict(img, device=f'cuda:{device}')
#
#
# # Paddle - ABC rec (only rec)
# padd_rec_abc = Rec_ABC.get_model(conf['padd_rec_model_dir'])
#
# # Paddle - small area rec
# padd_rec_sml = Rec_ALL.get_sml_model(conf['padd_rec_model_dir'])
#
# # Paddle - all text rec
# padd_rec_txt = Rec_ALL.get_model(conf['padd_rec_model_dir'])
#
# # 不包含尺寸元素的图纸类型（由主表类型界定）
# noSizeElemMainTypes = [
#     MainTableType.apartListMain,
#     MainTableType.apartDeviceMain,
#     MainTableType.listMain,
#     MainTableType.singleRowMain
# ]
#
#
# def small_area_text_rec(img, test=False, isDet=True):
#     def det_and_rec(image):
#         boxes, res, _ = padd_rec_sml(image)
#         if len(res) == 0:
#             return ''
#
#         if len(res) == 1:
#             return res[0][0]
#
#         resPack = sorted(list(zip(boxes, res)), key=lambda x: x[0][3][0])
#         sb = ''
#         for _, t in resPack:
#             sb += t[0]
#         return sb
#
#     def rec(image):
#         res, _ = padd_rec_abc([image])
#         return res[0][0]
#
#     if not isDet:
#         return rec(img)
#     text = det_and_rec(img)
#     # if test:
#     #     cv2.imwrite(
#     #         r'D:\Resources\Projects\Python\ultralytics-main-2\val\cell5' + f"\\[{text}]{comm.get_time_str()}.png", img)
#     if text.strip() == '':
#         height, width = img.shape[:2]
#         new_size = (int(width * 4), height)
#         stretched_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
#         text = det_and_rec(img)
#         if text.strip() == '':
#             return rec(stretched_img)
#     return text


# def process_image(file_contents: bytes):
#     # 将字节流转换为OpenCV图像
#     nparr = np.frombuffer(file_contents, np.uint8)
#     rgb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#     rgb_img_y = rgb_img.copy()
#
#     lower_yellow = np.array([0, 100, 100])
#     upper_yellow = np.array([30, 255, 255])
#
#     # 创建掩码识别黄色像素
#     yellow_mask = cv2.inRange(rgb_img_y, lower_yellow, upper_yellow)
#
#     # 识别出的黄色像素转为白色
#     rgb_img_y[yellow_mask > 0] = [255, 255, 255]
#
#     cv2.imwrite(r'D:\Resources\Projects\Python\ultralytics-main-2\val\noyellow' + f"\\{comm.get_time_str()}.png",
#                 rgb_img_y)
#
#     # 将图像转换为灰度图
#     gray_img = cv2.cvtColor(rgb_img_y, cv2.COLOR_BGR2GRAY)
#
#     # 二值化处理
#     thresh = 250
#     _, binary_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
#     binary_img_3c = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
#     cv2.imwrite(r'D:\Resources\Projects\Python\ultralytics-main-2\val\noyellow' + f"\\bin_{comm.get_time_str()}.png",
#                 binary_img_3c)
#     return binary_img_3c, rgb_img


def cv2_to_bytes(image):
    # 将OpenCV图像转换为字节流
    _, img_encoded = cv2.imencode('.png', image)
    return img_encoded.tobytes()


# pageIdSeparator = ','


# def predict_core(data_dict, img, index, pageId, raw_img):
#     data_dict.append_page(img, pageId)
#
#     # 检测 ABC 类
#     det_abc = Det_ABC(img)
#     det_abc.cut_img((720, 1280))
#     det_abc.predict([yolo_det_abc])
#     det_abc.nms()
#     # data_dict = det_abc.gen_dict()
#     data_dict.append_boxes(index, det_abc.get_boxes())
#
#     # 识别 ABC 类
#     # Rec_ABC.rec_abc(img, data_dict.get_page(index), padd_rec_abc)
#     Rec_ABC.rec_abc(raw_img, data_dict.get_page(index), padd_rec_abc)
#
#     # 检测 主表 / 明细表
#     det_shrink = Det_Shrink(img)
#     det_shrink.predict_table(yolo_det_table_pred, data_dict.get_page(index))
#
#     # 检测并识别 主表元素
#     DnR_mainTableElem = DetAndRecMainTableElem(img, data_dict.get_page(index), yolo_det_main_table_elem_pred,
#                                                small_area_text_rec, raw_img)
#
#     # 检测并识别 明细表元素
#     DetAndRecIndexTableElem(img, data_dict.get_page(index), yolo_det_index_table_elem_pred, small_area_text_rec,
#                             raw_img)
#
#     # 检测并识别 目录材料表元素
#     DetAndRecListTableElem(img, data_dict.get_page(index), yolo_det_list_table_elem_pred, small_area_text_rec, raw_img)
#
#     # 检测并识别 尺寸元素
#     if DnR_mainTableElem.getMainType() not in noSizeElemMainTypes:
#         # # 检测 非尺寸因素
#         # det_shrink.predict_tat(yolo_det_tat)
#
#         # 界定外边界
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         inner_box = None
#         try:
#             inner_box = TableExtractor(gray_img, img).get_inner_box_normalized()
#         except Exception:
#             inner_box = [0, 1, 1, 0]
#         finally:
#             # new
#             DetAndRecSizeElem(raw_img, data_dict.get_page(index), padd_rec_txt, yolo_det_tat_pred, inner_box)
#
#         # # 识别 所有文本元素
#         # # all_text_data = Rec_ALL.rec_all(img, padd_rec_txt)
#         # all_text_data = Rec_ALL.rec_all(raw_img, padd_rec_txt)
#
#         # # 筛选出尺寸
#         # det_shrink.filter_chicun(all_text_data, data_dict.get_page(index), inner_box)

#
# def download_file(bucket_name: str, file_name: str):
#     print(f'DOWNLOADING FILE: [{bucket_name}] [{file_name}]')
#     try:
#         data = minio_client.get_object(bucket_name, file_name)
#     except Exception as e:
#         print(f'[{bucket_name}] [{file_name}] DOWNLOAD_FILE_ERROR::{type(e).__name__}')
#         raise
#     else:
#         return data.read()


# @celery_app.task(name='zysd_server.predict_bt')
# def predict_bt(files, pageIdList, graphId, callbackUrl, bucketName):
#     data_dict = None
#     try:
#         pageIdList = pageIdList.split(pageIdSeparator)
#         data_dict = ResultFileGen(graphId)
#
#         for index, file_contents in enumerate(files):
#             if isinstance(file_contents, str):
#                 file_contents = download_file(bucketName, file_contents)
#             img, rgb_img = comm.process_image(comm.byte_to_img(file_contents))
#             predict_core(data_dict, img, index, pageIdList[index], rgb_img)
#             # await asyncio.get_event_loop().run_in_executor(executor, predict_core, data_dict, img, index,
#             #                                                pageIdList[index], rgb_img)
#
#         # 过滤无用字段
#         filter_field(data_dict.data)
#     except Exception as e:
#         data_dict.data['isSuccess'] = False
#         data_dict.data['errorInfo'] = traceback.format_exc()
#         print(data_dict.data['errorInfo'])
#
#     finally:
#         # 回调
#         callbackCnt = 0
#         with httpx.Client() as client:
#             while callbackCnt < callbackPatience:
#                 callbackCnt += 1
#                 try:
#                     start_time = time.time()
#                     response = client.post(callbackUrl, json=data_dict.data, timeout=10 + 20 * callbackCnt)
#                     # print('------RESPONSE--------')
#                     # print("Status code:", response.status_code)
#                     # print("Response JSON:", response.json())
#                     # print("Response text:", response.text)
#                     # print("Response content:", response.content)
#                     # print("Response headers:", response.headers)
#                 except (httpx.ConnectTimeout, httpcore.ConnectTimeout, httpx.ReadError, httpcore.ReadError,
#                         httpx.ReadTimeout, httpcore.ReadTimeout, httpx.WriteError, httpcore.WriteError,
#                         httpx.WriteTimeout, httpcore.WriteTimeout, httpx.RemoteProtocolError,
#                         httpcore.RemoteProtocolError):
#                     end_time = time.time()
#                     elapsed_time = end_time - start_time
#                     print(f"[duration: {elapsed_time:.2f}] [{graphId}] Retrying {callbackCnt}/{callbackPatience}...")
#                     if callbackCnt >= callbackPatience:
#                         print(f"[duration: {elapsed_time:.2f}] [{graphId}] Exceed Max Retrying Patience")
#                         raise
#                 except Exception as e:
#                     print(f"Httpx other error occurred: {type(e).__name__}")
#                     raise
#                 else:
#                     if response.status_code == 200:
#                         end_time = time.time()
#                         elapsed_time = end_time - start_time
#                         print(f'[duration: {elapsed_time:.2f}] [{graphId}] CALLBACK_SUCCESS____')
#                         return
#                 print('RE_CALLBACK____')
#                 time.sleep(random.randint(callbackCnt, 15))


@app.post('/[TEST]predict')
async def test_predict(files: list[UploadFile]):
    """

    Args:
        files: 支持上传多个文件（支持文件类型：pdf、zip压缩包、png等其他图片格式）

    Returns:
        包含关键标识标注的效果图和相关表格的HTML文件的压缩包（等待推理完毕后点击 Download file）

    """

    final_files = []
    for file in files:
        if file.filename.endswith('.zip'):
            file_bytes = io.BytesIO(await file.read())
            with zipfile.ZipFile(file_bytes, 'r') as zip_ref:
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_ref.extractall(temp_dir)
                    for file_path, file_name in fo.find_all_file_paths(temp_dir):
                        final_files.append({
                            'filename': file_name.encode('cp437').decode('gbk'),
                            'content': Path(file_path).read_bytes(),
                            'type': fo.get_suffix(file_name)
                        })
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
                # imgs, rgb_imgs = pdf2img.http_multi_page_pdf_convert_to_img(file_contents)
                tmp_rgb_imgs = pdf2img.http_multi_page_pdf_convert_to_img(file_contents)
                for t_img in tmp_rgb_imgs:
                    img, rgb_img = comm.process_image(t_img)
                    imgs.append(img)
                    rgb_imgs.append(rgb_img)
            else:
                img, rgb_img = comm.process_image(comm.byte_to_img(file_contents))
                imgs.append(img)
                rgb_imgs.append(rgb_img)

            j = 0
            for img, rgb_img in zip(imgs, rgb_imgs):
                j += 1
                prefix = f'{i}_{j}_'
                data_dict = ResultFileGen('TEST')
                html_list = []

                # TEST_only_classify_table(data_dict, img, 0, 'TEST', rgb_img)
                try:
                    # predict_core(data_dict, img, 0, 'TEST', rgb_img)
                    pageData = predict_test_page(img, rgb_img)
                    data_dict.appendPage(pageData)

                    # filter_field(data_dict.data, ['quad', 'confidence'])

                    draw_img = drawBox.drawBoxes(rgb_img, data_dict.data)

                    # html_list = []
                    html_list = [HTTPMainTableHTMLGen(data_dict.get_page(0)).html]
                    html_list += HTTPListTableHTMLGen(data_dict.get_page(0)).htmlList
                    html_list += HTTPIndexTableHTMLGen(data_dict.get_page(0)).htmlList
                except Exception as e:
                    error_info = traceback.format_exc()
                    zip_file.writestr(f'[ERROR]{prefix}{filename}.txt', error_info)
                    print(error_info)

                zip_file.writestr(f'{prefix}{filename}_image.png', cv2_to_bytes(draw_img))
                zip_file.writestr(f'{prefix}{filename}_data.json', json.dumps(data_dict.data, ensure_ascii=False, indent=4))

                k = 0

                for html in html_list:
                    if html is None:
                        continue
                    k += 1
                    zip_file.writestr(f'{prefix}{filename}_table[{k}].html', html)

    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, media_type='application/zip',
                             headers={"Content-Disposition": f"attachment; filename={quote(filename)}.zip"})


# def filter_field(data_dict, trivial_field=None):
#     if trivial_field is None:
#         trivial_field = ['quad', 'confidence', 'mainType', 'isApart', 'tableName']
#
#     for page in data_dict['pageData']:
#         for i in range(len(page['targets'])):
#             target = page['targets'][i]
#             for field in trivial_field:
#                 if field in target:
#                     target.pop(field)
#             # if 'quad' in target:
#             #     target.pop('quad')
#             # if 'confidence' in target:
#             #     target.pop('confidence')
#             # if 'mainType' in target:
#             #     target.pop('mainType')
#
#             # 去首尾空字符
#             if 'text' in target:
#                 text = target['text']
#                 if isinstance(text, str):
#                     target['text'] = text.strip()
#                 elif isinstance(text, list):  # 明细表/目录表
#                     for i in range(len(text)):
#                         for k, v in target['text'][i].items():
#                             if isinstance(v, str):
#                                 target['text'][i][k] = v.strip()
#                 elif isinstance(text, dict):  # 主表
#                     for k, v in target['text'].items():
#                         if isinstance(v, str):
#                             target['text'][k] = v.strip()
#
#             # 去“明细表/目录表”中空行数据项
#             if 'text' in target:
#                 text = target['text']
#                 if isinstance(text, list):  # 明细表/目录表
#                     for i in range(len(text) - 1, -1, -1):
#                         isEmptyRow = True
#                         for k, v in target['text'][i].items():
#                             if isinstance(v, str):
#                                 if len(v) > 0:
#                                     isEmptyRow = False
#                                     break
#                         if isEmptyRow:
#                             del target['text'][i]
#                             if 'cellBoxes' in target:
#                                 del target['cellBoxes'][i]


def main():
    import uvicorn
    uvicorn.run('zysd_server_test:app', host=conf['host'], port=conf['port']+1, reload=False, workers=conf['fastapi_workers'])


if __name__ == "__main__":
    main()
