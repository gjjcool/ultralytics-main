import copy
import sys

sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm
from tag_data import TargetTypeTag


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
    'page',  # 13
    'designPhase',
    'customerProjectNo',
    'companyProjectNo',
    'totalDesigner',
    'designMajor',
    'customerName',
    'projectName',
    'maiName',  # 21
    '[drawingList]',
    'customerDrawingName',  # 未对接
    'drawingName',
    '[materialTableList]'
]


class MainTableType:
    commonMain = 0
    listMain = 1
    apartListMain = 2
    apartDeviceMain = 3
    singleRowMain = 4


MainTableElemTagGroup = [
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
    {2, 3, 4, 6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
    {2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
    {9, 11, 12, 13, 23, 24},
    {11, 12, 13}
]

MainTableElemKeywordGroup = []
for group in MainTableElemTagGroup:
    keyGroup = set()
    for tag in group:
        keyGroup.add(tagMap[tag])
    MainTableElemKeywordGroup.append(keyGroup)


class MainTableTypeFlag:
    listMainFlag = 22
    apartListMainFlag = 25
    apartDeviceMainFlag = 24
    logo = 9
    drawingId = 11
    version = 12
    page = 13


class AnchorElemTag:
    rightmost = -1
    bottom = -2
    material = 8
    logo = 9
    listMainFlag = 22
    drawingName = 24
    apartListMainFlag = 25
    setDrawingName = 21
    chiefDesigner = 17
    departmentHead = 2
    designMajor = 18


# "titleType": "",//标题分类，0-材料表，1-设备列表，2-分图标题 3-目录标题
titleTypeCodeMap = {
    MainTableType.commonMain: 2,
    MainTableType.listMain: 3,
    MainTableType.singleRowMain: 3,
    MainTableType.apartListMain: 0,
    MainTableType.apartDeviceMain: 1
}

mainTableClsSet = {TargetTypeTag.mainTable,
                   TargetTypeTag.temp_apartListMainTable,
                   TargetTypeTag.temp_apartDeviceMainTable,
                   TargetTypeTag.temp_singleRowMainTable}
directRecTagSet = {9, 10, 11, 12, 13}
directRecKeywordSet = {tagMap[tag] for tag in directRecTagSet}
drawingIdTag = 11
noDetTagSet = {10, 11, 12, 13}

MainTableCellCount = []
for group in MainTableElemTagGroup:
    cnt = 0
    for tag in group:
        cnt += 1 if tag in directRecTagSet else 2
    MainTableCellCount.append(cnt)

predictHorizontalRule = {
    MainTableType.commonMain: {
        AnchorElemTag.logo: {1, 2, 3, 4, 5, 6, 7, 8},
        AnchorElemTag.material: {0}
    },
    MainTableType.listMain: {
        AnchorElemTag.logo: {15, 16, 17, 2, 3, 4, 6, 18},
        AnchorElemTag.rightmost: {19, 20},
        AnchorElemTag.listMainFlag: {21},
        AnchorElemTag.designMajor: {14}
    },
    MainTableType.apartDeviceMain: {
        AnchorElemTag.drawingName: {23},
        AnchorElemTag.rightmost: {24}
    },
    MainTableType.apartListMain: {
        AnchorElemTag.rightmost: {23, 15, 16},
        AnchorElemTag.apartListMainFlag: {21},
        AnchorElemTag.logo: {17, 2},
        AnchorElemTag.setDrawingName: {20, 19},
        AnchorElemTag.chiefDesigner: {14},
        AnchorElemTag.departmentHead: {18}
    }
}

predictVerticalRule = {
    MainTableType.apartListMain: {
        AnchorElemTag.bottom: {3, 4, 5, 6, 7}
    }
}

# 用于区分普通标题栏和目录标题栏
onlyBelong2CommMainTags = {0, 1, 5, 7, 8, 10}
onlyBelong2ListMainTags = {14, 15, 16, 17, 18, 19, 20, 21, 22}


def predictBox(mainType, cls, cand: comm.CandidateSet, candCopy: comm.CandidateSet):
    box = cand.getBox(cls)
    if mainType in predictHorizontalRule:
        for k, v in predictHorizontalRule[mainType].items():
            if (cls in v) and (candCopy.contains(k) or k < 0):
                x1 = candCopy.getBox(k)[0] if k > 0 else 1.0
                if x1 > box[2]:  # 避免截出来的图是 None
                    cand.setBox(cls, [box[2], box[1], x1, box[3]])

    if mainType in predictVerticalRule:
        for k, v in predictVerticalRule[mainType].items():
            if (cls in v) and (candCopy.contains(k) or k < 0):
                y1 = candCopy.getBox(k)[1] if k > 0 else 1.0
                if y1 > box[3]:  # 避免截出来的图是 None
                    cand.setBox(cls, [box[0], box[3], box[2], y1])


def judgeTitleType(pageData, mainType):
    if mainType == MainTableType.apartDeviceMain:
        for t in pageData['targets']:
            if t['class'] == TargetTypeTag.listTable:
                return 0

    return titleTypeCodeMap[mainType]


def calcSearchRatio(text: dict, mainType: int):
    correctCellCnt = 0
    for k in text.keys():
        if k in MainTableElemKeywordGroup[mainType]:
            correctCellCnt += 1 if k in directRecKeywordSet else 2
    return correctCellCnt / MainTableCellCount[mainType]


class DetAndRecMainTableElem:
    def __init__(self, img, pageData, detModel, recModel, raw_img):
        self.__detModel = detModel
        self.__recModel = recModel
        self.__mainType = None

        # 选择置信度最高的标题栏（保证一张图标题栏的唯一性）
        conf = 0.0
        mainTarget = None
        for t in pageData['targets']:
            if t['class'] in mainTableClsSet:
                if t['confidence'] > conf:
                    conf = t['confidence']
                    mainTarget = t

        # 删除置信度较低的标题栏候选框
        for i in range(len(pageData['targets']) - 1, -1, -1):
            t = pageData['targets'][i]
            if t['class'] in mainTableClsSet and t['confidence'] < conf:
                del pageData['targets'][i]

        if mainTarget is not None:
            tableImg = comm.extractImg(img, mainTarget['box'])
            rawTableImg = comm.extractImg(raw_img, mainTarget['box'])
            rawTableImg = comm.remove_table_frame_lines(rawTableImg, tableImg)
            self.__mainType, boxes = self.__classify(tableImg, mainTarget['class'])
            mainTarget['mainType'] = self.__mainType  # for gen html
            mainTarget['text'], mainTarget['cellBoxes'] = self.__predictElem(rawTableImg, mainTarget['box'],
                                                                             self.__mainType, boxes)
            mainTarget['searchRatio'] = calcSearchRatio(mainTarget['text'], self.__mainType)
            mainTarget['text']['titleType'] = judgeTitleType(pageData, self.__mainType)
            mainTarget['class'] = TargetTypeTag.mainTable

    def __classify(self, tableImg, tableCls):
        boxes = self.__detModel(tableImg)[0].boxes

        if tableCls == TargetTypeTag.temp_apartDeviceMainTable:
            return MainTableType.apartDeviceMain, boxes
        if tableCls == TargetTypeTag.temp_singleRowMainTable:
            return MainTableType.singleRowMain, boxes
        if tableCls == TargetTypeTag.temp_apartListMainTable:
            return MainTableType.apartListMain, boxes

        hasLogo = False
        # hasGreater13Tag = False
        logoTopPos = None
        drawingIdMidY = None
        versionBox = None
        pageBox = None
        commScore = 0
        listScore = 0

        for box, cls, _ in comm.zipBoxes(boxes, tableImg):
            if cls == MainTableTypeFlag.listMainFlag:
                return MainTableType.listMain, boxes

            if cls == MainTableTypeFlag.apartListMainFlag:
                return MainTableType.apartListMain, boxes

            if cls == MainTableTypeFlag.apartDeviceMainFlag:
                return MainTableType.apartDeviceMain, boxes

            if cls == MainTableTypeFlag.logo:
                hasLogo = True
                logoTopPos = box[1]

            if cls == MainTableTypeFlag.drawingId:
                drawingIdMidY = (box[1] + box[3]) / 2

            if cls == MainTableTypeFlag.version:
                versionBox = box

            if cls == MainTableTypeFlag.page:
                pageBox = box

            if cls in onlyBelong2CommMainTags:
                commScore += 1
            elif cls in onlyBelong2ListMainTags:
                listScore += 1

            # if cls > 13:
            #     hasGreater13Tag = True

        # if len(boxes.xyxy.tolist()) < 5:  # 元素较少判定为单行标题栏
        #     return MainTableType.singleRowMain, boxes
        # if (not hasLogo) and len(boxes.xyxy.tolist()) < 5:  # 没有logo元素并且元素较少判定为单行标题栏
        #     return MainTableType.singleRowMain, boxes
        #
        # if hasLogo and logoTopPos >= 0.5:  # logo在下半方则判定为分离-目录标题栏
        #     return MainTableType.apartListMain, boxes
        #
        # if drawingIdMidY:  # 图号和（版次、页码）不在同一水平线则判定为分离-设备标题栏
        #     if versionBox and (not (versionBox[1] < drawingIdMidY < versionBox[3])):
        #         return MainTableType.apartDeviceMain, boxes
        #     if pageBox and (not (pageBox[1] < drawingIdMidY < pageBox[3])):
        #         return MainTableType.apartDeviceMain, boxes

        # if hasGreater13Tag:  # （在普通标题栏和目录标题栏之间判断）若含大于13号的标签则判定为目录标题栏
        #     return MainTableType.listMain, boxes
        if listScore > commScore:  # 检测到的非共有元素哪边多就判定为那类表格
            return MainTableType.listMain, boxes

        return MainTableType.commonMain, boxes

    def __predictElem(self, tableImg, tableBox, mainType, boxes):
        textRes = dict()
        boxRes = dict()

        cand = comm.CandidateSet()
        candCopy = comm.CandidateSet()

        # 候选：同 tag 对象选取置信度高者
        for box, cls, conf in comm.zipBoxes(boxes, img=tableImg):
            cand.add(cls, box, conf)
            candCopy.add(cls, box, conf)

        # 推理字段值位置：在 cand 中原地修改，candCopy 副本用于记录 anchor 元素的原始位置
        # candCopy = copy.deepcopy(cand)
        for key in candCopy.keys():  # TEST
            print(f'[{key}]: {candCopy.getBox(key)}')

        for cls in cand.data.keys():
            if cls not in directRecTagSet:
                predictBox(mainType, cls, cand, candCopy)

        # 识别对应位置文字并记录内容和位置信息
        for cls in cand.data.keys():
            if tagMap[cls][0] == '[':  # 可忽略的字段
                continue

            subImg = comm.extractImg(tableImg, cand.getBox(cls))
            if cls == drawingIdTag:
                text = comm.postProcessDrawingId(
                    self.__recModel(comm.preProcessDrawingIdImg(subImg), isDet=False)
                )
            elif cls in noDetTagSet:
                text = self.__recModel(subImg, isDet=False)
            else:
                text = self.__recModel(subImg)
            textRes[tagMap[cls]] = text
            boxRes[tagMap[cls]] = comm.getCellBox(tableBox, cand.getBox(cls))

        return textRes, boxRes

    def getMainType(self):
        return self.__mainType
