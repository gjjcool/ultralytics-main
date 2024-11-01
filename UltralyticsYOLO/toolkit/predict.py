import concurrent.futures
import json
import os
import threading
import time
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# import torch.multiprocessing as mp
import sys
from queue import Queue

sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm


def normalize(box, h, w):
    box[0] /= w
    box[2] /= w
    box[1] /= h
    box[3] /= h


def genPageData(img, graphPageId: str):
    img_h, img_w = img.shape[:2]
    pageData = dict()
    pageData['imageHeight'] = img_h
    pageData['imageWidth'] = img_w
    pageData['graphPageId'] = graphPageId
    pageData['targets'] = []
    pageData['isSuccess'] = True
    pageData['errorInfo'] = ''

    return pageData


def pageAppendBoxes(pageData, boxes):
    img_h = pageData['imageHeight']
    img_w = pageData['imageWidth']

    for box in boxes:
        normalize(box['xyxy'], img_h, img_w)
        b = box['xyxy']
        quad = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        pageData['targets'].append({
            'signId': '',
            'toSignId': '',
            'toGraphNo': '',
            'box': [[b[0], b[1]], [b[2], b[3]]],
            'quad': quad,
            'class': int(box['cls']),
            'confidence': box['conf'],
            'text': ''
        })


class ResultFileGen:
    def __init__(self, graphId: str):
        self.data = dict()
        self.data['graphId'] = graphId
        self.data['pageData'] = []
        self.data['isSuccess'] = True
        self.data['errorInfo'] = ''

        # self.data['imageHeight'] = img_h
        # self.data['imageWidth'] = img_w
        # self.data['targets'] = []

        # for box in boxes:
        #     normalize(box['xyxy'], img_h, img_w)
        #     b = box['xyxy']
        #     quad = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        #     pageData['targets'].append({
        #         'signId': '',
        #         'toSignId': '',
        #         'toGraphNo': '',
        #         'box': [[b[0], b[1]], [b[2], b[3]]],
        #         'quad': quad,
        #         'class': int(box['cls']),
        #         'confidence': box['conf'],
        #         'text': ''
        #     })

    def get_page(self, index):
        return self.data['pageData'][index]

    def getPageNum(self):
        return len(self.data['pageData'])

    def appendPage(self, pageData):
        self.data['pageData'].append(pageData)

    def append_page(self, img, graphPageId: str):
        img_h, img_w = img.shape[:2]
        pageData = dict()
        pageData['imageHeight'] = img_h
        pageData['imageWidth'] = img_w
        pageData['graphPageId'] = graphPageId
        pageData['targets'] = []
        self.data['pageData'].append(pageData)

    def append_boxes(self, index, boxes):
        img_h = self.data['pageData'][index]['imageHeight']
        img_w = self.data['pageData'][index]['imageWidth']

        for box in boxes:
            normalize(box['xyxy'], img_h, img_w)
            b = box['xyxy']
            quad = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
            self.data['pageData'][index]['targets'].append({
                'signId': '',
                'toSignId': '',
                'toGraphNo': '',
                'box': [[b[0], b[1]], [b[2], b[3]]],
                'quad': quad,
                'class': int(box['cls']),
                'confidence': box['conf'],
                'text': ''
            })

    def save(self, file_dir, file_name):
        with open(os.path.join(file_dir, file_name + '.json'), 'w', encoding='utf-8') as json_file:
            json.dump(self.data, json_file, indent=2)


# lock = threading.Lock()

def split_dict_evenly(input_dict, n_parts):
    # 将字典的项转为列表
    items = list(input_dict.items())

    # 计算每部分的大小
    chunk_size = len(items) // n_parts
    remainder = len(items) % n_parts

    # 生成平分后的字典列表
    chunks = []
    start = 0

    for i in range(n_parts):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk = items[start:end]
        chunks.append(chunk)
        start = end

    return chunks


def threadPredict(model, chunk):
    res = []
    for offset, img in chunk:
        res.append((offset, model(img)[0].boxes))

    return res


class Predictor:
    def __init__(self, img):
        self.__img = img
        self.H, self.W = self.__img.shape[:2]
        self.__sub_imgs = dict()
        self.__boxes = list()
        self.pred_duration = 0
        self.__nms_duration = 0
        # self.__executor = ThreadPoolExecutor(max_workers=3)

    def cut_img(self, size: tuple[int, int], step: tuple[int, int] = None):
        h, w = size
        if step is None:
            step = (h // 2, w // 2)

        for y in range(0, self.H, step[0]):
            for x in range(0, self.W, step[1]):
                sub_img = self.__img[y:min(y + h, self.H), x:min(x + w, self.W)]
                self.__sub_imgs[(y, x)] = sub_img

    def predict(self, models):
        def recordBoxes(offset, boxes):
            if len(boxes.xyxy.tolist()) > 0:
                for box, cls, conf in comm.zipBoxes(boxes):
                    self.__boxes.append({
                        'box': [box[1] + offset[0], box[0] + offset[1], box[3] + offset[0], box[2] + offset[1]],  # yxyx
                        'xyxy': [box[0] + offset[1], box[1] + offset[0], box[2] + offset[1], box[3] + offset[0]],
                        'conf': conf,
                        'cls': cls,
                        'reserve': True
                    })

        st = time.time()
        # offsetList = list(self.__sub_imgs.keys())
        # for i in range(0, len(offsetList), 2):
        #     future0 = self.__executor.submit(model, self.__sub_imgs[offsetList[i]], i)
        #     future1 = None if i + 1 >= len(offsetList) \
        #         else self.__executor.submit(model, self.__sub_imgs[offsetList[i + 1]], i + 1)
        #
        #     boxes0 = future0.result()[0].boxes
        #     boxes1 = None if future1 is None else future1.result()[0].boxes
        #
        #     recordBoxes(offsetList[i], boxes0)
        #     if boxes1 is not None:
        #         recordBoxes(offsetList[i + 1], boxes1)

        # data_queue = Queue()
        # for offset, img in self.__sub_imgs.items():
        #     data_queue.put((offset, img))
        # chunks = split_dict_evenly(self.__sub_imgs, len(models))
        #
        # res_list = []
        # # with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # with ProcessPoolExecutor(max_workers=len(models)) as executor:
        #     futures = [executor.submit(threadPredict, model, chunk) for model, chunk in zip(models, chunks)]
        #     for future in concurrent.futures.as_completed(futures):
        #         res_list += future.result()
        #
        # for offset, boxes in res_list:
        #     recordBoxes(offset, boxes)

        for offset, img in self.__sub_imgs.items():
            boxes = models[0](img)[0].boxes
            if len(boxes.xyxy.tolist()) > 0:
                recordBoxes(offset, boxes)

        self.pred_duration = time.time() - st

    def iou(self, a: list, b: list, iou_thresh, iob_thresh=0.8):
        y_overlap = min(a[2], b[2]) - max(a[0], b[0])
        x_overlap = min(a[3], b[3]) - max(a[1], b[1])
        if y_overlap <= 0 or x_overlap <= 0:
            return 0
        inter = x_overlap * y_overlap
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        union = (a[2] - a[0]) * (a[3] - a[1]) + b_area - inter

        # 交并比大于阈值，或者 A 区域较大 B 较小，A 几乎包含 B 的情况
        return inter / union >= iou_thresh or inter / b_area >= iob_thresh

    def nms(self, thresh=0.3):
        """非极大值抑制"""
        st = time.time()

        boxes = []
        self.__boxes = sorted(self.__boxes, key=lambda x: x['conf'], reverse=True)
        for i in range(len(self.__boxes)):
            box = self.__boxes[i]
            if box['reserve']:
                boxes.append(box)
                for j in range(i + 1, len(self.__boxes)):
                    other = self.__boxes[j]
                    if other['reserve'] and self.iou(box['box'], other['box'], thresh):
                        self.__boxes[j]['reserve'] = False
        self.__boxes = boxes

        self.__nms_duration = time.time() - st

    def draw_img(self):
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for box in self.__boxes:
            beg = (int(box['xyxy'][0]), int(box['xyxy'][1]))
            end = (int(box['xyxy'][2]), int(box['xyxy'][3]))
            cv2.rectangle(self.__img, beg, end, palette[int(box['cls'])], 2)

        return self.__img

    # def gen_json(self, img_name, save_dir):
    #     stem = os.path.basename(img_name).split('.')[0]
    #     ResultFileGen(self.__boxes, self.H, self.W, img_name, self.__pred_duration, self.__nms_duration).save(save_dir, stem)

    # def gen_dict(self):
    #     return ResultFileGen(self.__boxes, self.H, self.W, '', 0, 0).data

    def get_boxes(self):
        return self.__boxes

    def get_all_boxes_normalized(self):
        boxes = []
        for box in self.__boxes:
            xyxy = box['xyxy']
            xyxy[0] /= self.W
            xyxy[2] /= self.W
            xyxy[1] /= self.H
            xyxy[3] /= self.H
            boxes.append(xyxy)
        return boxes
