import sys
sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm

tagMap = [
    'proportion',
    'quality',
    'deptHead',
    'chiefEngineer',
    'chiefDesigner',
    'audit',
    'designer',
    'cartography',
    'material',
    'logo',
    'drawingName',
    'graphNo',
    'version',
    'page'
]


class DetAndRecMainTableElem:
    def __init__(self, img, pageData, detModel, recModel):
        self.__curLogoLeft = -1  # 当前Logo左边界
        self.__curMatLeft = -1  # 当前材质左边界
        for t in pageData['targets']:
            if t['class'] == 3:
                tableImg = comm.extractImg(img, t['box'])
                t['text'], t['cellBoxes'] = self.detAndRecElem(tableImg, detModel, recModel, t['box'])

    def detAndRecElem(self, tableImg, detModel, recModel, tableBox):
        res = dict()
        boxRes = dict()

        res['titleType'] = 2

        h, w = tableImg.shape[:2]
        self.__curLogoLeft, self.__curMatLeft = -1, -1
        candidates = dict()
        boxes = detModel(tableImg)[0].boxes
        if len(boxes.xyxy.tolist()) > 0:
            for box, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                comm.normalize(box, h, w)
                if cls == 8 and comm.updateCandidates(candidates, cls, box, conf):
                    self.__curMatLeft = box[0]
                elif cls == 9 and comm.updateCandidates(candidates, cls, box, conf):
                    self.__curLogoLeft = box[0]

            for box, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                comm.normalize(box, h, w)
                self.__judgeElem(res, tableImg, box, cls, conf, recModel, boxRes, tableBox, candidates)

            for k, v in candidates.items():
                img = comm.extractImg(tableImg, v['box'])
                if k == 11:  # 图号
                    img = comm.preProcessDrawingIdImg(img)
                text = recModel(img)
                if k == 11:  # 图号
                    text = comm.postProcessDrawingId(text)
                res[tagMap[k]] = text
                boxRes[tagMap[k]] = comm.getCellBox(tableBox, v['box'])

        return res, boxRes

    def __judgeElem(self, res, tableImg, box, cls, conf, recModel, boxRes, tableBox, candidates):
        """
            对于框即是要识别的值，直接记录该 box 位置；
            对于框是键值，则根据“材料”和“Logo”框的位置额外推理出其值的框 valBox 位置。
        """
        cls = int(cls)

        if cls >= 9:
            comm.updateCandidates(candidates, cls, box, conf)

            # res[tagMap[cls]] = recModel(comm.extractImg(tableImg, box))
            # boxRes[tagMap[cls]] = comm.getCellBox(tableBox, box)

        elif cls == 0:
            if self.__curMatLeft < box[2]:
                print(f'ERROR::self.__curMatLeft[{self.__curMatLeft}]')
                # res[tagMap[cls]] = ''
                return
            valBox = [box[2], box[1], self.__curMatLeft, box[3]]
            comm.updateCandidates(candidates, cls, valBox, conf)

            # res[tagMap[cls]] = recModel(comm.extractImg(tableImg, valBox))
            # boxRes[tagMap[cls]] = comm.getCellBox(tableBox, valBox)

        else:  # 8 or 1 (其他的手写内容暂未考虑)
            if self.__curLogoLeft < box[2]:
                print(f'ERROR::self.__curLogoLeft[{self.__curLogoLeft}]')
                # res[tagMap[cls]] = ''
                return
            valBox = [box[2], box[1], self.__curLogoLeft, box[3]]
            comm.updateCandidates(candidates, cls, valBox, conf)

            # res[tagMap[cls]] = recModel(comm.extractImg(tableImg, valBox))
            # boxRes[tagMap[cls]] = comm.getCellBox(tableBox, valBox)
