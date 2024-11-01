import json
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
# sys.path.insert(0, parent_dir)
# # sys.path.append('celery_apps')
# sys.path.append('../UltralyticsYOLO/toolkit')

from ultralytics import YOLO
from UltralyticsYOLO.toolkit.predict import Predictor as Det_ABC
from UltralyticsYOLO.toolkit.predict import ResultFileGen, genPageData, pageAppendBoxes
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
from minio import Minio
import cv2

with open('zysd_server_conf.json', 'r', encoding='utf-8') as conf_file:
    conf = json.load(conf_file)


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
padd_rec_abc = Rec_ABC.get_model(conf["padd_rec_model_dir"])

# Paddle - small area rec (检测+识别)
padd_rec_sml = Rec_ALL.get_sml_model(conf["padd_rec_model_dir"])

# Paddle - all text rec (检测+识别)
padd_rec_txt = Rec_ALL.get_model(conf["padd_rec_model_dir"])


# endregion

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


minio_clients = {}
def initialize_minio_clients():
    global minio_clients

    # 获取测试和生产环境的配置
    minio_configs = {
        'test': {
            'endpoint': '10.108.22.5:9000',
            'access_key': 'minioadmin',
            'secret_key': 'minioadmin',
            'bucket_name': 'cisdi-drawing',
            'secure': False
        },
        'prod': {
            'endpoint': '10.108.22.31:9000',
            'access_key': 'avZbKXU48LIrYNc8i7Fd',
            'secret_key': 'tt6hmZMc0IY9tikL0PZC3pSz10Lx2w21yeVIX6x7',
            'bucket_name': 'cisdi-drawing',
            'secure': False
        }
    }

    # 创建 MinIO 客户端
    minio_clients['test'] = Minio(
        endpoint=minio_configs['test']['endpoint'],
        access_key=minio_configs['test']['access_key'],
        secret_key=minio_configs['test']['secret_key'],
        secure=minio_configs['test']['secure']
    )
    minio_clients['prod'] = Minio(
        endpoint=minio_configs['prod']['endpoint'],
        access_key=minio_configs['prod']['access_key'],
        secret_key=minio_configs['prod']['secret_key'],
        secure=minio_configs['prod']['secure']
    )

initialize_minio_clients()

def get_minio_client(host):
    # 根据 host 返回相应的 MinIO 客户端
    if '10.108.22.7' in host:
        return minio_clients['test']
    else:
        return minio_clients['prod']

# 用于请求图纸文件
#minio_client = Minio(
#   endpoint='10.108.22.5:9000',
#    access_key='minioadmin',
#   secret_key='minioadmin',
#    secure=False
#)



def predict_page(filename, pageId, bucketName, host):
    file_contents = download_file(bucketName, filename, host)
    img, raw_img = comm.process_image(comm.byte_to_img(file_contents))
    pageData = genPageData(img, pageId)
    predict_core(img, raw_img, pageData)
    filter_field(pageData)
    return pageData


def predict_test_page(img, raw_img):
    pageData = genPageData(img, 'TEST')
    predict_core(img, raw_img, pageData)
    filter_field(pageData)
    return pageData


def predict_core(img, raw_img, pageData):
    # 检测 ABC 类
    det_abc = Det_ABC(img)
    det_abc.cut_img((720, 1280))
    det_abc.predict(yolo_det_abc_models)
    det_abc.nms()
    # data_dict = det_abc.gen_dict()
    # data_dict.append_boxes(index, det_abc.get_boxes())
    pageAppendBoxes(pageData, det_abc.get_boxes())

    # 识别 ABC 类
    # Rec_ABC.rec_abc(img, data_dict.get_page(index), padd_rec_abc)
    Rec_ABC.rec_abc(raw_img, pageData, padd_rec_abc)

    # 检测 主表 / 明细表
    det_shrink = Det_Shrink(img)
    det_shrink.predict_table(yolo_det_table_pred, pageData)

    # 检测并识别 主表元素
    DnR_mainTableElem = DetAndRecMainTableElem(img, pageData, yolo_det_main_table_elem_pred,
                                               small_area_text_rec, raw_img)

    # 检测并识别 明细表元素
    DetAndRecIndexTableElem(img, pageData, yolo_det_index_table_elem_pred, small_area_text_rec,
                            raw_img)

    # 检测并识别 目录材料表元素
    DetAndRecListTableElem(img, pageData, yolo_det_list_table_elem_pred, small_area_text_rec, raw_img)

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
            DetAndRecSizeElem(raw_img, pageData, padd_rec_txt, yolo_det_tat_pred, inner_box)

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
        new_size = (int(width * 2), height)
        stretched_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        text = det_and_rec( stretched_img)
        if text.strip() == '':
            return rec(stretched_img)
    return text


def download_file(bucket_name: str, file_name: str, host):
    # 获取对应环境的 MinIO 客户端
    minio_client = get_minio_client(host)
    print(f'DOWNLOADING FILE: [{bucket_name}] [{file_name}]')
    try:
        data = minio_client.get_object(bucket_name, file_name)
    except Exception as e:
        print(f'[{bucket_name}] [{file_name}] DOWNLOAD_FILE_ERROR::{type(e).__name__}')
        raise
    else:
        return data.read()


def filter_field(page, trivial_field=None):
    if trivial_field is None:
        trivial_field = ['quad', 'confidence', 'tableName']
        # trivial_field = ['quad', 'confidence', 'mainType', 'isApart', 'tableName']

    # for page in data_dict['pageData']:
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
                    for k, v in target['text'][i].items()::
                        if isinstance(v, str) and k != 'totalMass':
                            if len(v) > 0:
                                isEmptyRow = False
                                break
                    if isEmptyRow:
                        del target['text'][i]
                        if 'cellBoxes' in target:
                            del target['cellBoxes'][i]

        #识别“旧”
        if 'text' in target:
            text = target['text']
            if isinstance(text, list):  # 明细表/目录表
                for i in range(len(text)):
                    for k, v in text[i].items():
                        if isinstance(v, str) and "旧" in v:
                            text[i][k] = "旧"
                    # 识别“M”并转为“3”
                        if k == 'pages' and "M" in v:
                            text[i][k] = v.replace("M", "3")
                    # 如果内容为空格字符或者空字符，则直接删除此数据项
                    if v.strip() == "":
                        del text[i][k]