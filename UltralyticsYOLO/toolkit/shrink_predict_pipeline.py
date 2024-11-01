import copy
import sys
import cv2
sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm
from tag_data import TargetTypeTag


def normalize(box, h, w):
    box[0] /= w
    box[2] /= w
    box[1] /= h
    box[3] /= h


def calc_overlap_area_of_two_rect(a, b):
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
    return x_overlap * y_overlap


def is_chicun(box, pageData, tat_boxes, in_box):
    if box[1][1] <= in_box[0] or box[0][0] >= in_box[1] or box[0][1] >= in_box[2] or box[1][0] <= in_box[3]:
        return False

    thresh_iou = 0.5
    thresh_contain = 0.9
    box_area = (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])

    for target in pageData['targets']:
        tb = target['box']
        t_area = (tb[1][0] - tb[0][0]) * (tb[1][1] - tb[0][1])
        inter_area = calc_overlap_area_of_two_rect(tb, box)
        iou_rat = inter_area / (t_area + box_area - inter_area)
        if iou_rat >= thresh_iou:
            return False
        contain_rat = inter_area / box_area
        if contain_rat >= thresh_contain:
            return False

    for tb in tat_boxes:
        t_area = (tb[1][0] - tb[0][0]) * (tb[1][1] - tb[0][1])
        inter_area = calc_overlap_area_of_two_rect(tb, box)
        iou_rat = inter_area / (t_area + box_area - inter_area)
        if iou_rat >= thresh_iou:
            return False
        contain_rat = inter_area / box_area
        if contain_rat >= thresh_contain:
            return False

    return True


# table_class_map = [
#     3, 4, 3, 6, 10, 11, 12, 13
# ]

table_class_map = [
    TargetTypeTag.mainTable,
    TargetTypeTag.indexTable,
    TargetTypeTag.temp_singleRowMainTable,
    TargetTypeTag.listTable,
    TargetTypeTag.temp_apartListMainTable,
    TargetTypeTag.temp_apartDeviceMainTable,
    TargetTypeTag.indexTable
]


class PredictPipeline:
    def __init__(self, img, img_size=640):
        self.classes = None
        self.bboxes = None
        self.tat_boxes = []
        self.__img = img
        self.__sml_img = None
        self.__shrink_rate = 1
        self.target_size = None

        h, w = self.__img.shape[:2]
        if h >= w:
            self.__shrink_rate = h / img_size
            self.target_size = (int(w // self.__shrink_rate), int(img_size))
            self.__sml_img = cv2.resize(self.__img, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            self.__shrink_rate = w / img_size
            self.target_size = (int(img_size), int(h // self.__shrink_rate))
            self.__sml_img = cv2.resize(self.__img, self.target_size, interpolation=cv2.INTER_AREA)

    def get_tables_on_raw_img(self):
        main_table_imgs = []
        index_table_imgs = []

        for i in range(len(self.bboxes)):
            for j in range(4):
                self.bboxes[i][j] = int(self.bboxes[i][j] * self.__shrink_rate)

        for i in range(len(self.classes)):
            bb = self.bboxes[i]
            table_img = self.__img[bb[1]:bb[3], bb[0]:bb[2]]
            if int(self.classes[i]) == 0:
                main_table_imgs.append(table_img)
            elif int(self.classes[i]) == 1:
                index_table_imgs.append(table_img)

        return main_table_imgs, index_table_imgs

    def get_sml_img(self):
        return self.__sml_img

    def predict_table(self, model, pageData):
        results = model(self.__sml_img)
        boxes = results[0].boxes
        self.bboxes = results[0].boxes.xyxy.tolist()
        self.classes = results[0].boxes.cls.tolist()
        for box, cls, conf in comm.zipBoxes(boxes):
            normalize(box, self.target_size[1], self.target_size[0])
            pageData['targets'].append({
                'box': [[box[0], box[1]], [box[2], box[3]]],
                # 'class': 3 + int(cls) + (1 if int(cls) > 1 else 0),  # 0/1=>3/4; 2/3=>6/7
                'class': table_class_map[int(cls)],
                'modelClass': int(cls),  # new code
                'confidence': conf,
            })

    def predict_tat(self, model):
        results = model(self.__sml_img)
        for xyxy in results[0].boxes.xyxy.tolist():
            normalize(xyxy, self.target_size[1], self.target_size[0])
            self.tat_boxes.append([[xyxy[0], xyxy[1]], [xyxy[2], xyxy[3]]])

    def filter_chicun(self, all_text_data, pageData, inner_box):
        tmp_pageData = copy.deepcopy(pageData)
        for inf_box in all_text_data:
            quad = inf_box['quad']
            x0 = min(quad[0][0], quad[1][0], quad[2][0], quad[3][0])
            y0 = min(quad[0][1], quad[1][1], quad[2][1], quad[3][1])
            x1 = max(quad[0][0], quad[1][0], quad[2][0], quad[3][0])
            y1 = max(quad[0][1], quad[1][1], quad[2][1], quad[3][1])
            box = [[x0, y0], [x1, y1]]
            if is_chicun(box, tmp_pageData, self.tat_boxes, inner_box):
                pageData['targets'].append({
                    'box': box,
                    'class': 5,
                    'text': inf_box['text']
                })


