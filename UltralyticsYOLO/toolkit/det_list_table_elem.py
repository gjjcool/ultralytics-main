import sys

import cv2
import numpy as np

sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm

tagMap = [
    'serialNumber',
    'graphName',
    'graphNo',
    'version',
    'pages',
    'newOld',
    'zeHe',
    'remark',
    '[number]',  # 8
    'total',
    'matTabNo',
    '[reuse]',
    'notice'
]


def judgeApartTable(boxes):
    for _, cls, _ in comm.zipBoxes(boxes):
        if cls == 5 or cls == 6:
            return False
    return True


class DetAndRecListTableElem:
    def __init__(self, img, pageData, detModel, recModel, raw_img):
        self.__header = dict()
        self.__pageData = pageData

        for t in pageData['targets']:
            if t['class'] == 6:
                tableImg = comm.extractImg(img, t['box'])
                rawTableImg = comm.extractImg(raw_img, t['box'])
                rawTableImg = comm.remove_table_frame_lines(rawTableImg, tableImg)
                t['text'], t['cellBoxes'], t['isApart'] = self.detAndRecElem(tableImg, detModel, recModel, t['box'],
                                                                             rawTableImg)

    def detAndRecElem(self, tableImg, detModel, recModel, tableBox, rawTableImg):
        res = []
        boxRes = []

        boxes = detModel(tableImg)[0].boxes

        isApartTable = judgeApartTable(boxes)

        # 先记录表头位置信息
        self.__recordHeader(boxes, tableImg)

        # 定位序号列所在的 x 轴位置
        numColPosX = self.__locateNumColPosX()

        for box, cls, conf in comm.zipBoxes(boxes, img=tableImg):
            # comm.normalizeByImg(box, tableImg)
            # cls = int(cls)

            # 表头中未检测到“序号”时，处理所有的 cls=8 的候选框
            if cls == 8 and (numColPosX < 0 or box[0] <= numColPosX <= box[2]):
                data = dict()
                posData = dict()
                data[tagMap[0]] = recModel(comm.extractImg(rawTableImg, box))
                posData[tagMap[0]] = comm.getCellBox(tableBox, box)
                for k, v in self.__header.items():
                    if k == 0:
                        continue

                    valBox = [v['box'][0], box[1], v['box'][2], box[3]]
                    if k == 2:  # 图号
                        data[tagMap[k]] = comm.postProcessDrawingId(
                            recModel(comm.preProcessDrawingIdImg(comm.extractImg(rawTableImg, valBox)))
                        )
                        posData[tagMap[k]] = comm.getCellBox(tableBox, valBox)
                    else:
                        data[tagMap[k]] = recModel(comm.extractImg(rawTableImg, valBox))
                        posData[tagMap[k]] = comm.getCellBox(tableBox, valBox)
                res.append(data)
                boxRes.append(posData)

            # 材料表图号 / 施工图号
            elif cls == 10:
                if 2 in self.__header:
                    valBox = [self.__header[2]['box'][0], box[1], self.__header[2]['box'][2], box[3]]
                    drawingId = comm.postProcessDrawingId(
                        recModel(comm.preProcessDrawingIdImg(comm.extractImg(rawTableImg, valBox)))
                    )
                    self.__pageData['targets'].append({
                        'box': comm.getCellBox(tableBox, valBox),
                        'class': 8 if isApartTable else 7,
                        'text': drawingId
                    })

        res, boxRes = comm.sortMultiItems(res, boxRes)

        return res, boxRes, isApartTable

    def __recordHeader(self, boxes, img):
        info = img.shape
        width = info[1]
        lines = comm.detect_straight_lines(img)
        isProcess = lines is not None and lines.any()
        if isProcess:
            d = comm.process_lines_biaoge(lines, img)
        for box, cls, conf in comm.zipBoxes(boxes, img=img):
            if cls <= 7:
                if (cls not in self.__header) or (conf > self.__header[cls]['conf']):  # 保留置信度高的表头元素
                    if isProcess:
                        box = comm.adjust_box(box, d, width)
                    self.__header[cls] = {'box': box, 'conf': conf}

    def __locateNumColPosX(self):
        if 0 in self.__header:
            serialNumberBox = self.__header[0]['box']
            return (serialNumberBox[0] + serialNumberBox[2]) / 2
        return -1  # TODO: 表头中未检测到“序号”
