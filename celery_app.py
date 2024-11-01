if __name__ == '__main__':
    import eventlet
    eventlet.monkey_patch()

import threading
import json
import os
import random
import time

import cv2
from minio import Minio
import httpx
import httpcore
import traceback
from celery import Celery
# from celery.loaders.app import AppLoader

from UltralyticsYOLO.toolkit.predict import Predictor as Det_ABC
from UltralyticsYOLO.toolkit.predict import ResultFileGen
from UltralyticsYOLO.toolkit.shrink_predict_pipeline import PredictPipeline as Det_Shrink
from UltralyticsYOLO.toolkit.table_extractor_draw import TableExtractor
from UltralyticsYOLO.toolkit.det_main_table_elem import DetAndRecMainTableElem
from UltralyticsYOLO.toolkit.det_index_table_elem import DetAndRecIndexTableElem
from UltralyticsYOLO.toolkit.det_list_table_elem import DetAndRecListTableElem
from UltralyticsYOLO.toolkit.det_size_elem import DetAndRecSizeElem
from UltralyticsYOLO.toolkit.det_main_table_elem import MainTableType

import PaddleOCR.rec_abc as Rec_ABC
import PaddleOCR.rec_all as Rec_ALL

import UltralyticsYOLO.toolkit.comm_toolkit as comm

import multiprocessing as mp


# region YOLO Model Predict Functions
# def yolo_det_abc_pred(img, index):
#     return (yolo_det_abc if index % 2 == 0 else yolo_det_abc_2).predict(img)
# return yolo_det_abc.predict(img)


def yolo_det_table_pred(img):
    return yolo_det_table.predict(img)


def yolo_det_main_table_elem_pred(img):
    return yolo_det_main_table_elem.predict(img)


def yolo_det_index_table_elem_pred(img):
    return yolo_det_index_table_elem.predict(img)


def yolo_det_list_table_elem_pred(img):
    return yolo_det_list_table_elem.predict(img)


def yolo_det_tat_pred(img):
    return yolo_det_tat.predict(img)


# endregion

# 不包含尺寸元素的图纸类型（由主表类型界定）
noSizeElemMainTypes = [
    MainTableType.apartListMain,
    MainTableType.apartDeviceMain,
    MainTableType.listMain,
    MainTableType.singleRowMain
]

# 用于请求图纸文件
minio_client = Minio(
    endpoint='10.108.22.5:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

pageIdSeparator = ','
callbackPatience = 10
callbackInterval = 5

celery_app = Celery('celery_app', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')


@celery_app.task
def predict_bt(files, pageIdList, graphId, callbackUrl, bucketName):
    data_dict = None
    try:
        pageIdList = pageIdList.split(pageIdSeparator)
        data_dict = ResultFileGen(graphId)

        for index, file_contents in enumerate(files):
            if isinstance(file_contents, str):
                file_contents = download_file(bucketName, file_contents)
            img, rgb_img = comm.process_image(comm.byte_to_img(file_contents))
            predict_core(data_dict, img, index, pageIdList[index], rgb_img)
            # await asyncio.get_event_loop().run_in_executor(executor, predict_core, data_dict, img, index,
            #                                                pageIdList[index], rgb_img)

        # 过滤无用字段
        filter_field(data_dict.data)
    except Exception as e:
        data_dict.data['isSuccess'] = False
        data_dict.data['errorInfo'] = traceback.format_exc()
        print(data_dict.data['errorInfo'])

    finally:
        # 回调
        callbackCnt = 0
        with httpx.Client() as client:
            while callbackCnt < callbackPatience:
                callbackCnt += 1
                try:
                    start_time = time.time()
                    response = client.post(callbackUrl, json=data_dict.data, timeout=10 + 20 * callbackCnt)
                    # print('------RESPONSE--------')
                    # print("Status code:", response.status_code)
                    # print("Response JSON:", response.json())
                    # print("Response text:", response.text)
                    # print("Response content:", response.content)
                    # print("Response headers:", response.headers)
                except (httpx.ConnectTimeout, httpcore.ConnectTimeout, httpx.ReadError, httpcore.ReadError,
                        httpx.ReadTimeout, httpcore.ReadTimeout, httpx.WriteError, httpcore.WriteError,
                        httpx.WriteTimeout, httpcore.WriteTimeout, httpx.RemoteProtocolError,
                        httpcore.RemoteProtocolError):
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"[duration: {elapsed_time:.2f}] [{graphId}] Retrying {callbackCnt}/{callbackPatience}...")
                    if callbackCnt >= callbackPatience:
                        print(f"[duration: {elapsed_time:.2f}] [{graphId}] Exceed Max Retrying Patience")
                        raise
                except Exception as e:
                    print(f"Httpx other error occurred: {type(e).__name__}")
                    raise
                else:
                    if response.status_code == 200:
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f'[duration: {elapsed_time:.2f}] [{graphId}] CALLBACK_SUCCESS____')
                        return
                print('RE_CALLBACK____')
                time.sleep(random.randint(callbackCnt, 15))


def predict_core(data_dict, img, index, pageId, raw_img):
    data_dict.append_page(img, pageId)

    # 检测 ABC 类
    det_abc = Det_ABC(img)
    det_abc.cut_img((720, 1280))
    det_abc.predict(yolo_det_abc_models)
    det_abc.nms()
    # data_dict = det_abc.gen_dict()
    data_dict.append_boxes(index, det_abc.get_boxes())

    # 识别 ABC 类
    # Rec_ABC.rec_abc(img, data_dict.get_page(index), padd_rec_abc)
    Rec_ABC.rec_abc(raw_img, data_dict.get_page(index), padd_rec_abc)

    # 检测 主表 / 明细表
    det_shrink = Det_Shrink(img)
    det_shrink.predict_table(yolo_det_table_pred, data_dict.get_page(index))

    # 检测并识别 主表元素
    DnR_mainTableElem = DetAndRecMainTableElem(img, data_dict.get_page(index), yolo_det_main_table_elem_pred,
                                               small_area_text_rec, raw_img)

    # 检测并识别 明细表元素
    DetAndRecIndexTableElem(img, data_dict.get_page(index), yolo_det_index_table_elem_pred, small_area_text_rec,
                            raw_img)

    # 检测并识别 目录材料表元素
    DetAndRecListTableElem(img, data_dict.get_page(index), yolo_det_list_table_elem_pred, small_area_text_rec, raw_img)

    # 检测并识别 尺寸元素
    if DnR_mainTableElem.getMainType() not in noSizeElemMainTypes:
        # # 检测 非尺寸因素
        # det_shrink.predict_tat(yolo_det_tat)

        # 界定外边界
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inner_box = None
        try:
            inner_box = TableExtractor(gray_img, img).get_inner_box_normalized()
        except Exception:
            inner_box = [0, 1, 1, 0]
        finally:
            # new
            DetAndRecSizeElem(raw_img, data_dict.get_page(index), padd_rec_txt, yolo_det_tat_pred, inner_box)

        # # 识别 所有文本元素
        # # all_text_data = Rec_ALL.rec_all(img, padd_rec_txt)
        # all_text_data = Rec_ALL.rec_all(raw_img, padd_rec_txt)

        # # 筛选出尺寸
        # det_shrink.filter_chicun(all_text_data, data_dict.get_page(index), inner_box)

    print(f'PREDICT DURATION: [ABC={det_abc.pred_duration:.2f}s]')


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
    # if test:
    #     cv2.imwrite(
    #         r'D:\Resources\Projects\Python\ultralytics-main-2\val\cell5' + f"\\[{text}]{comm.get_time_str()}.png", img)
    if text.strip() == '':
        height, width = img.shape[:2]
        new_size = (int(width * 4), height)
        stretched_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        text = det_and_rec(img)
        if text.strip() == '':
            return rec(stretched_img)
    return text


def download_file(bucket_name: str, file_name: str):
    print(f'DOWNLOADING FILE: [{bucket_name}] [{file_name}]')
    try:
        data = minio_client.get_object(bucket_name, file_name)
    except Exception as e:
        print(f'[{bucket_name}] [{file_name}] DOWNLOAD_FILE_ERROR::{type(e).__name__}')
        raise
    else:
        return data.read()


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


print('OUT_MAIN')
if __name__ == '__main__':
    from ultralytics import YOLO
    print('MAIN')
    with open('zysd_server_conf.json', 'r', encoding='utf-8') as conf_file:
        conf = json.load(conf_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


    # region YOLO Models
    def init_models(models: list, modelSign):
        for cudaInd, num in conf[f'{modelSign}_info'].items():
            for _ in range(num):
                models.append(YOLO(conf[modelSign]).to(f'cuda:{cudaInd}'))


    # YOLO - ABC det
    # yolo_det_abc = (YOLO(conf["yolo_det_abc"])
    #                 .to(f'cuda:{conf["yolo_det_abc_device"]}'))
    # yolo_det_abc_2 = (YOLO(conf["yolo_det_abc"])
    #                   .to(f'cuda:{conf["yolo_det_abc_2_device"]}'))
    yolo_det_abc_models = []
    init_models(yolo_det_abc_models, 'yolo_det_abc')

    # YOLO - table(main/index/id/list) det
    yolo_det_table = (YOLO(conf["yolo_det_table"])
                      .to(f'cuda:{conf["yolo_det_table_device"]}'))

    # YOLO - main table elem det
    yolo_det_main_table_elem = (YOLO(conf["yolo_det_main_table_elem"])
                                .to(f'cuda:{conf["yolo_det_main_table_elem_device"]}'))

    # YOLO - index table elem det
    yolo_det_index_table_elem = (YOLO(conf["yolo_det_index_table_elem"])
                                 .to(f'cuda:{conf["yolo_det_index_table_elem_device"]}'))

    # YOLO - list table elem det
    yolo_det_list_table_elem = (YOLO(conf["yolo_det_list_table_elem"])
                                .to(f'cuda:{conf["yolo_det_list_table_elem_device"]}'))

    # YOLO - non-size(text and table) det
    yolo_det_tat = (YOLO(conf["yolo_det_tat"])
                    .to(f'cuda:{conf["yolo_det_tat_device"]}'))

    # endregion

    # region Paddle Models
    # Paddle - ABC rec (只识别不检测)
    padd_rec_abc = Rec_ABC.get_model()

    # Paddle - small area rec (检测+识别)
    padd_rec_sml = Rec_ALL.get_sml_model()

    # Paddle - all text rec (检测+识别)
    padd_rec_txt = Rec_ALL.get_model()
    # endregion

    # mp.set_start_method('spawn')
    # celery_app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=1', '--logfile=/root/drawing_rec/ultralytics-main-2/celery_log.log'])
    def start_worker(i):
        print(f'--hostname=worker{i+1}@%h')
        celery_app.worker_main(
            argv=['worker', '--loglevel=info', f'--concurrency={conf["celery_concurrency"]}',
                  f'--pool={conf["celery_pool"]}', f'--hostname=worker{i+1}@%h'])

    threads = []
    for i in range(conf['celery_workers']):
        thread = threading.Thread(target=start_worker, args=(i,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
