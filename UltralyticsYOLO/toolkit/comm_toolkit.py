import copy
import re
from datetime import datetime

import numpy as np

import cv2


def extractImg(img, box):
    box = copy.deepcopy(box)
    h, w = img.shape[:2]

    def func(img, box, isRatio):
        if len(box) == 4:  # xyxy
            if isRatio:
                box[0] = int(box[0] * w)
                box[1] = int(box[1] * h)
                box[2] = int(box[2] * w)
                box[3] = int(box[3] * h)
            else:
                box[0] = int(box[0])
                box[1] = int(box[1])
                box[2] = int(box[2])
                box[3] = int(box[3])

            if box[1] >= box[3] or box[0] >= box[2]:
                return img[0:1, 0:1]
            return img[box[1]:box[3], box[0]:box[2]]

        if len(box) == 2:  # [[xy][xy]]
            if isRatio:
                box[0][0] = int(box[0][0] * w)
                box[0][1] = int(box[0][1] * h)
                box[1][0] = int(box[1][0] * w)
                box[1][1] = int(box[1][1] * h)
            else:
                box[0][0] = int(box[0][0])
                box[0][1] = int(box[0][1])
                box[1][0] = int(box[1][0])
                box[1][1] = int(box[1][1])

            if box[0][1] >= box[1][1] or box[0][0] >= box[1][0]:
                return img[0:1, 0:1]
            return img[box[0][1]:box[1][1], box[0][0]:box[1][0]]

    isRatio = False
    if len(box) == 4 and box[3] <= 1:
        isRatio = True
    elif len(box) == 2 and box[0][1] <= 1:
        isRatio = True
    return func(img, box, isRatio)


def box_4x1_to_box_2x2(box):
    # [xyxy] = [1,2,3,4]
    # [[xy],[xy]] = [[1,2], [3,4]]
    if len(box) != 4:
        raise ValueError(f'box_4x1_to_box_2x2:: len(box) is not expected.')
    return [[box[0], box[1]], [box[2], box[3]]]


def normalize(box, h, w):
    if len(box) == 4:
        box[0] /= w
        box[2] /= w
        box[1] /= h
        box[3] /= h
    elif len(box) == 2:
        box[0][0] /= w
        box[1][0] /= w
        box[0][1] /= h
        box[1][1] /= h
    return box


def normalizeByImg(box, img):
    h, w = img.shape[:2]
    return normalize(box, h, w)


def getAbsBoxByImg(box, img):
    b = copy.deepcopy(box)
    h, w = img.shape[:2]
    b[0][0] = int(b[0][0] * w)
    b[1][0] = int(b[1][0] * w)
    b[0][1] = int(b[0][1] * h)
    b[1][1] = int(b[1][1] * h)
    return b


def getCellBox(tableBox, box):
    cellBox = [[0, 0], [0, 0]]
    w = abs(tableBox[1][0] - tableBox[0][0])
    h = abs(tableBox[1][1] - tableBox[0][1])
    cellBox[0][0] = tableBox[0][0] + w * box[0]
    cellBox[1][0] = tableBox[0][0] + w * box[2]
    cellBox[0][1] = tableBox[0][1] + h * box[1]
    cellBox[1][1] = tableBox[0][1] + h * box[3]
    return cellBox


class CandidateSet:
    def __init__(self):
        self.data = dict()

    def add(self, cls, box, conf):
        cls = int(cls)
        if (cls not in self.data) or conf > self.data[cls]['conf']:
            self.data[cls] = {'box': box, 'conf': conf}

    def setBox(self, cls, box):
        if not self.contains(cls):
            return

        self.data[cls]['box'] = box

    def getBox(self, cls):
        if not self.contains(cls):
            return [0.999, 0.999, 1.0, 1.0]  #

        return self.data[cls]['box']

    def contains(self, cls):
        return cls in self.data

    def keys(self):
        return self.data.keys()


def updateCandidates(candidates: dict, cls, box, conf):
    cls = int(cls)
    if (cls not in candidates) or conf > candidates[cls]['conf']:
        candidates[cls] = {'box': box, 'conf': conf}
        return True
    return False


def postProcessDrawingId(s):
    """
    去除字符串首尾的非阿拉伯数字和非字母字符
    """
    return re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', s)


def preProcessDrawingIdImg(image):
    height, width = image.shape[:2]
    new_size = (int(width * 1.7), height)
    stretched_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return stretched_image


def zipBoxes(boxes, img=None, imgSize=None):  # imgSize = (h, w)
    """

    Args:
        boxes:
        img:
        imgSize:

    Returns:
        zip(box, cls, conf)

    """

    def isValidBox(box):
        return box[2] - box[0] > 1 and box[3] - box[1] > 1

    xyxyList = boxes.xyxy.tolist()
    xyxyList = [box for box in xyxyList if isValidBox(box)]

    if img is not None:
        xyxyList = [normalizeByImg(box, img) for box in xyxyList]
    elif imgSize is not None:
        xyxyList = [normalize(box, imgSize[0], imgSize[1]) for box in xyxyList]

    clsList = boxes.cls.tolist()
    clsList = [int(cls) for cls in clsList]

    return zip(xyxyList, clsList, boxes.conf.tolist())


def sortMultiItems(res, boxRes):
    if len(res) == 0:
        return res, boxRes

    # 根据 box 位置从上到下排序数据项信息
    combined = list(zip(res, boxRes))
    combined_sorted = sorted(combined, key=lambda x: x[1]['serialNumber'][0][1])
    sorted_res, sorted_boxRes = zip(*combined_sorted)
    return list(sorted_res), list(sorted_boxRes)


def calc_overlap_area_of_two_rect(a, b):
    if len(a) == 2:
        if a[0][0] > a[1][0]:
            a = [a[1], a[0]]
        if b[0][0] > b[1][0]:
            b = [b[1], b[0]]

        x_overlap = min(a[1][0], b[1][0]) - max(a[0][0], b[0][0])
        if x_overlap <= 0:
            return 0
        y_overlap = min(a[1][1], b[1][1]) - max(a[0][1], b[0][1])
        if y_overlap <= 0:
            return 0

    elif len(a) == 4:
        if a[0] > a[2]:
            a = [a[2], a[3], a[0], a[1]]
        if b[0] > b[2]:
            b = [b[2], b[3], b[0], b[1]]
        x_overlap = min(a[2], b[2]) - max(a[0], b[0])
        if x_overlap <= 0:
            return 0
        y_overlap = min(a[3], b[3]) - max(a[1], b[1])
        if y_overlap <= 0:
            return 0

    else:
        raise ValueError(f'len(a) is not expected.')

    return x_overlap * y_overlap


def calc_iou(a, b):
    if len(a) == 4:
        a_area = abs(a[2] - a[0]) * abs(a[3] - a[1])
        b_area = abs(b[2] - b[0]) * abs(b[3] - b[1])
    elif len(a) == 2:
        a_area = abs(a[1][0] - a[0][0]) * abs(a[1][1] - a[0][1])
        b_area = abs(b[1][0] - b[0][0]) * abs(b[1][1] - b[0][1])
    else:
        raise ValueError(f'len(a) is not expected.')
    inter_area = calc_overlap_area_of_two_rect(a, b)
    if (a_area + b_area - inter_area) == 0:
        return 1.0
    return inter_area / (a_area + b_area - inter_area)


def is_area_contain(a, b, thresh=0.8):
    if len(a) == 4:
        a_area = abs(a[2] - a[0]) * abs(a[3] - a[1])
        b_area = abs(b[2] - b[0]) * abs(b[3] - b[1])
    elif len(a) == 2:
        a_area = abs(a[1][0] - a[0][0]) * abs(a[1][1] - a[0][1])
        b_area = abs(b[1][0] - b[0][0]) * abs(b[1][1] - b[0][1])
    else:
        raise ValueError(f'len(a) is not expected.')
    inter_area = calc_overlap_area_of_two_rect(a, b)
    if a_area == 0 or b_area == 0:
        return False
    return inter_area / a_area >= thresh or inter_area / b_area >= thresh


def calc_iot(truth, pred):
    if len(truth) == 4:
        t_area = abs(truth[2] - truth[0]) * abs(truth[3] - truth[1])
        p_area = abs(pred[2] - pred[0]) * abs(pred[3] - pred[1])
    elif len(truth) == 2:
        t_area = abs(truth[1][0] - truth[0][0]) * abs(truth[1][1] - truth[0][1])
        p_area = abs(pred[1][0] - pred[0][0]) * abs(pred[1][1] - pred[0][1])
    else:
        raise ValueError(f'len(truth) is not expected.')
    if t_area == 0:
        return 1.0
    inter_area = calc_overlap_area_of_two_rect(truth, pred)
    return inter_area / t_area


def get_time_str():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    return timestamp


def remove_table_frame_lines(rawImg, binImg):
    # return img
    rawImg = copy.deepcopy(rawImg)
    binImg = copy.deepcopy(binImg)
    gray = cv2.cvtColor(binImg, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # cv2.imwrite(r'D:\Resources\Projects\Python\ultralytics-main-2\val\noyellow' + f"\\bin_inv_{get_time_str()}.png",
    #             cv2.cvtColor(copy.deepcopy(binary), cv2.COLOR_GRAY2BGR))

    # 检测水平和垂直线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # 合并水平和垂直线
    # lines = cv2.add(horizontal_lines, vertical_lines)

    # cv2.imwrite(r'D:\Resources\Projects\Python\ultralytics-main-2\val\png\test.png', lines)

    # 进一步过滤长直线
    min_line_length = 100  # 最小线段长度
    lines_h = cv2.HoughLinesP(horizontal_lines, 1, np.pi / 180, threshold=100, minLineLength=min_line_length,
                              maxLineGap=5)
    lines_v = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, threshold=100, minLineLength=min_line_length,
                              maxLineGap=5)
    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            cv2.line(rawImg, (x1, y1), (x2, y2), (255, 255, 255), 3)
    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            cv2.line(rawImg, (x1, y1), (x2, y2), (255, 255, 255), 3)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # test
    # cv2.imwrite(r'D:\Resources\Projects\Python\ultralytics-main-2\val' + f"\\{get_time_str()}.png", img)

    return rawImg


def byte_to_img(file_contents: bytes):
    nparr = np.frombuffer(file_contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def process_image(rgb_img):
    # # 将字节流转换为OpenCV图像
    # nparr = np.frombuffer(file_contents, np.uint8)
    # rgb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rgb_img_y = rgb_img.copy()

    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 创建掩码识别黄色像素
    yellow_mask = cv2.inRange(rgb_img_y, lower_yellow, upper_yellow)

    # 识别出的黄色像素转为白色
    rgb_img_y[yellow_mask > 0] = [255, 255, 255]

    #cv2.imwrite(r'C:\Users\15420\Desktop\123' + f"\\{get_time_str()}.png",
    #            rgb_img_y)

    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(rgb_img_y, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    thresh = 250
    _, binary_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)

    binary_img_3c = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    #cv2.imwrite(r'C:\Users\15420\Desktop\123' + f"\\bin_{get_time_str()}.png",
    #            binary_img_3c)

    return binary_img_3c, rgb_img


# in：   img
# out：  直线（线段）的端点横纵坐标（x1,x2,y1,y2）
def detect_straight_lines(img):
    # 转换成灰度图
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 边缘检测, Sobel算子大小为3
    edges = cv2.Canny(image_gray, 170, 220, apertureSize=3)  # 170 220
    # 霍夫曼直线检测，寻找像素大于50(一格)的直线/精度为1像素，低于5像素的线段会自动连接
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=50, maxLineGap=10)

    return lines


# in：   lines(x1,x2,y1,y2) , img
# out：  边框横坐标（单元素列表）
# def:    处理表格pdf（大）
def process_lines_biaoge(lines, img):
    d = []
    f = []
    g = []
    e = []
    info = img.shape
    height = info[0]
    width = info[1]
    for line in lines:
        # 获取所有符合条件直线坐标
        x1, y1, x2, y2 = line[0]
        if (x1 == x2):
            f.append([x1, x2, y1, y2])
            g.append(x1)
        # 获取列的信息
        if (x1 == x2) & (abs(y1 - y2) >= 0.5 * height):
            d.append(x1)
    # 不连续直线处理
    print(f)
    print(g)
    for i in range(len(f) - 1):
        error = 0
        counts = np.bincount(g)
        lin = np.argmax(counts)
        #        print(lin)
        for i in range(len(f) - 1):
            if (abs(lin - f[i][1]) < 5):
                error = error + abs(f[i][2] - f[i][3])
        #        print("error" , error,0.25*height)
        if (error >= 0.25 * height):
            d.append(lin)
        for j in range(max(counts)):
            g.remove(lin)
        if (max(counts) <= 2):
            print(max(counts))
            break

    print(d)
    print("biaoge")
    d.append(width)
    d.append(0)
    d.sort()
    d = list(set(d))
    d.sort()
    print(d)
    return d

# in：   lines(x1,x2,y1,y2) , img
# out：  边框横坐标（单元素列表）
# def：   处理图纸pdf（小）
def process_lines_tuzhi(lines, img):
    d = []
    f = []
    y = []  # 寻找上基准线
    lin = 0  # 基准线
    info = img.shape
    height = info[0]
    width = info[1]
    for line in lines:
        # 获取所有符合条件直线坐标
        x1, y1, x2, y2 = line[0]
        # 获取列的信息
        if (x1 == x2) & (abs(y1 - y2) >= 50):
            f.append([x1, x2, y1, y2])
            y.append(y2)
    # 获取众数，求得基准上线
    y.sort()
    y = y[2:-2]
    counts = np.bincount(y)
    lin = np.argmax(counts)

    for i in range(len(f)):
        print(f[i])
        if (abs(f[i][3] - lin) < 6):
            print(f[i])
            d.append(f[i][0])
    print("tuzhi")
    d.append(width)
    d.append(0)
    d.sort()
    d = list(set(d))
    d.sort()
    print(d)

    return d

# in：   box(框两点坐标) , d（真框横坐标）, width（图片的宽度）
# out：  调整后的box
def adjust_box(box, d, width):
    # 调整参考坐标比例，向左移动
    x = ((box[0] + box[2]) * 0.5) * width
    # 添加偏中心坐标和边缘坐标
    d.append(x)
    # 排序所有长度大于0.9*height像素的边框直线升序
    d.sort()
    # 选出中心左边两边的坐标并更改BOX的两点的横坐标
    for i in range(0, (len(d) - 1)):
        if (d[i] == x):
            # 归一化处理
            box[0] = (d[i - 1] / width)
            box[2] = (d[i + 1] / width)
            d.pop(i)
    return box


#if __name__ == "__main__":
#    remove_table_frame_lines(
#        cv2.imread(r'C:\Users\Lenovo\Desktop\ultralytics-main-2-update\val\png\[raw]20240731220254176706.png'))
