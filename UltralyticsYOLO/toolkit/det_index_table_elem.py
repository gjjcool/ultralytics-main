import sys

import cv2
import numpy as np

sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm

# from collections import OrderedDict

tagMap = [
    'serialNumber',
    'deviceNo',
    'deviceName',
    'deviceNum',
    'material',
    'singlePiece',
    'total',
    'remark',
    'totalMass',
    '[number]'
]


class PseudoYOLOBox:
    class PseudoList:
        def __init__(self, data):
            self.data = data

        def tolist(self):
            return self.data

    def __init__(self, boxes):
        self.xyxy = self.PseudoList([b['box'] for b in boxes])
        self.cls = self.PseudoList([b['cls'] for b in boxes])
        self.conf = self.PseudoList([b['conf'] for b in boxes])


class DetAndRecIndexTableElem:
    def __init__(self, img, pageData, detModel, recModel, raw_img):
        self.__hasDetHeader = False  # 是否已经检测到表头了
        self.__noHeaderTables = []  # 待检测的不具有表头的表格 (tableImg, boxes, target, tableBox)
        self.__header = dict()
        self.__totalMassValue = None
        self.__raw_img = raw_img
        self.__is_rec_test = False
        self.__graph_type = 0
        for t in pageData['targets']:
            if t['class'] == 4:
                if t['modelClass'] == 6:  # 表格类型（分离表格）
                    self.__graph_type = 1
                tableImg = comm.extractImg(img, t['box'])
                t['text'], t['cellBoxes'], t['tableName'] = self.detAndRecElem(tableImg, detModel, recModel, t,
                                                                               t['box'], None)

        if self.__hasDetHeader:
            for nht in self.__noHeaderTables:
                nht[2]['text'], nht[2]['cellBoxes'], nht[2]['tableName'] = self.detAndRecElem(nht[0], detModel,
                                                                                              recModel,
                                                                                              nht[2], nht[3], nht[1])

    def __containsHeader(self, boxes):
        minCnt = 3
        cnt = 0  # 来自表头单元格检测框的数量
        for cls in boxes.cls.tolist():
            if cls < 9:
                cnt += 1
            if cnt >= minCnt:
                return True
        return False

    def __recordHeader(self, boxes, img):
        info = img.shape
        width = info[1]
        lines = comm.detect_straight_lines(img)
        isProcess = lines is not None and lines.any()
        if isProcess:
            if self.__graph_type == 0:
                d = comm.process_lines_tuzhi(lines, img)
            if self.__graph_type == 1:
                d = comm.process_lines_biaoge(lines, img)
        for box, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            cls = int(cls)
            if cls <= 7:  # 8
                comm.normalizeByImg(box, img)
                if (cls not in self.__header) or (conf > self.__header[cls]['conf']):  # 保留置信度高的表头元素
                    if isProcess:
                        box = comm.adjust_box(box, d, width)
                    self.__header[cls] = {'box': box, 'conf': conf}
            if cls == 8:  # 8针对质量这一个box的处理，即什么也不做
                comm.normalizeByImg(box, img)
                if (cls not in self.__header) or (conf > self.__header[cls]['conf']):  # 保留置信度高的表头元素
                    self.__header[cls] = {'box': box, 'conf': conf}

    def __locateNumColPosX(self, boxes):
        posBoxCnt = dict()  # key: 候选框竖直基准线，val: 处于该基准线周围的“序号”候选框数量
        for box, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
            if cls == 9:
                flag = False  # posBoxCnt 中是否有穿过该候选框的基准线
                for k in posBoxCnt.keys():
                    if box[0] <= k <= box[2]:
                        posBoxCnt[k] += 1
                        flag = True
                        break
                if not flag:
                    x = (box[0] + box[2]) / 2
                    posBoxCnt[x] = 1
        keys = sorted(posBoxCnt.keys())
        maxCnt = 0
        pos = 0
        for k in keys:
            if posBoxCnt[k] > maxCnt:  # 保留检测框最多的那一列，若数量相同则优先取靠左边的
                maxCnt = posBoxCnt[k]
                pos = k
        return pos

    def __predictBox(self, tableImg, detModel, threshHeight=3000, cutHeight=2000, cutStep=1500):
        tableH, tableW = tableImg.shape[:2]
        # if tableH < threshHeight:
        #     return detModel(tableImg)[0].boxes

        if tableH >= threshHeight:
            # 切割
            sub_imgs = dict()
            for y in range(tableH - 1, -1, -cutStep):
                isLast = False
                if y - cutHeight < 0:
                    isLast = True
                    y = cutHeight
                sub_img = tableImg[y - cutHeight:y, 0:tableW]
                sub_imgs[y - cutHeight] = sub_img
                if isLast:
                    break
        else:
            sub_imgs = {0: tableImg}

        # 推理子图
        allBoxes = []
        for offsetY, img in sub_imgs.items():
            boxes = detModel(img)[0].boxes
            if len(boxes.xyxy.tolist()) > 0:
                for box, cls, conf in comm.zipBoxes(boxes):
                    allBoxes.append({
                        'box': [box[0], box[1] + offsetY, box[2], box[3] + offsetY],
                        'conf': conf,
                        'cls': cls,
                        'reserve': True
                    })

        # 非极大值抑制
        iou_thresh = 0.3
        iot_thresh = 0.5
        rsvBoxes = []
        allBoxes = sorted(allBoxes, key=lambda x: x['conf'], reverse=True)
        for i in range(len(allBoxes)):
            box = allBoxes[i]
            if box['reserve']:
                rsvBoxes.append(box)
                for j in range(i + 1, len(allBoxes)):
                    other = allBoxes[j]
                    if other['reserve'] and (
                            comm.calc_iou(box['box'], other['box']) >= iou_thresh or comm.calc_iot(other['box'], box[
                        'box']) >= iot_thresh):
                        allBoxes[j]['reserve'] = False

        return PseudoYOLOBox(rsvBoxes)

    def detAndRecElem(self, tableImg, detModel, recModel, target, tableBox, boxes):
        res = []
        boxRes = []
        tableName = ''

        # boxes 为空是首次推理；若不为空则是之前被寄存的没有表头的表
        if boxes is None:
            boxes = self.__predictBox(tableImg, detModel)

        # 若目前没检测到表头并且该表格不包含表头，则将其先寄存，后续再推理
        if (not self.__hasDetHeader) and (not self.__containsHeader(boxes)):
            self.__noHeaderTables.append((tableImg, boxes, target, tableBox))
            return res, boxRes, tableName

        # 若目前没检测到表头并且该表格包含表头，则先标记表头元素
        if not self.__hasDetHeader:
            self.__recordHeader(boxes, tableImg)
            self.__hasDetHeader = True

        rawTableImg = comm.extractImg(self.__raw_img, tableBox)  # 在原图上识别文字
        tableImg = comm.remove_table_frame_lines(rawTableImg, tableImg)  # 去框线
        numColPosX = self.__locateNumColPosX(boxes)
        # box xyxy
        for box, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
            if cls == 9 and box[0] <= numColPosX <= box[2]:
                data = dict()
                posData = dict()
                comm.normalizeByImg(box, tableImg)
                data[tagMap[0]] = recModel(comm.extractImg(tableImg, box), self.__is_rec_test)
                posData[tagMap[0]] = comm.getCellBox(tableBox, box)
                for k, v in self.__header.items():
                    if k == 0:
                        continue

                    if k == 8:  # 总质量（每个表唯一）
                        if self.__totalMassValue is None:
                            valBox = [v['box'][2], v['box'][1], 1.0, v['box'][3]]
                            self.__totalMassValue = recModel(comm.extractImg(tableImg, valBox), self.__is_rec_test)
                            posData[tagMap[k]] = comm.getCellBox(tableBox, valBox)
                        data[tagMap[k]] = self.__totalMassValue
                    elif k == 1:  # 代号/图号
                        valBox = [v['box'][0], box[1], v['box'][2], box[3]]
                        data[tagMap[k]] = recModel(comm.preProcessDrawingIdImg(comm.extractImg(tableImg, valBox)),
                                     self.__is_rec_test)
                        posData[tagMap[k]] = comm.getCellBox(tableBox, valBox)
                    else:
                        valBox = [v['box'][0], box[1], v['box'][2], box[3]]
                        data[tagMap[k]] = recModel(comm.extractImg(tableImg, valBox), self.__is_rec_test)
                        posData[tagMap[k]] = comm.getCellBox(tableBox, valBox)
                res.append(data)
                boxRes.append(posData)

            if cls == 10:
                tableName = recModel(comm.extractImg(tableImg, box), self.__is_rec_test)

        # 根据 box 位置从上到下排序数据项信息
        res, boxRes = comm.sortMultiItems(res, boxRes)

        return res, boxRes, tableName
