class TargetTypeTag:
    leadWire = 0
    arrowAndFlag = 1
    sectionalDrawingFlag = 2
    mainTable = 3
    indexTable = 4
    sizeFlag = 5
    listTable = 6
    materialTableDrawingId = 7  # 材料表图号
    workingDrawingId = 8  # 施工图图号

    temp_apartListMainTable = 10
    temp_apartDeviceMainTable = 11
    temp_apartDeviceIndexTable = 12

    temp_singleRowMainTable = 13

    nameList = [
        '引线', '剖断面', '剖视图', '标题栏', '明细表', '尺寸',
        '目录材料表', '材料表图号', '施工图图号'
    ]


class TitleTypeTag:
    materialTable = 0  # 分离的目录的标题栏（包括非首页的小表）
    deviceTable = 1
    commonTable = 2
    listTable = 3


