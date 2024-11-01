import copy
import sys

import cv2

sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm
from tag_data import TargetTypeTag

def is_size_elem(box, pageData, tat_boxes, in_box):
    if box[1][1] <= in_box[0] or box[0][0] >= in_box[1] or box[0][1] >= in_box[2] or box[1][0] <= in_box[3]:
        return False

    thresh_iou = 0.5
    thresh_contain = 0.9
    box_area = (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])

    for target in pageData['targets']:
        tb = target['box']
        t_area = (tb[1][0] - tb[0][0]) * (tb[1][1] - tb[0][1])
        inter_area = comm.calc_overlap_area_of_two_rect(tb, box)
        iou_rat = inter_area / (t_area + box_area - inter_area)
        if iou_rat >= thresh_iou:
            return False
        contain_rat = inter_area / box_area
        if contain_rat >= thresh_contain:
            return False

    for tb in tat_boxes:
        t_area = (tb[1][0] - tb[0][0]) * (tb[1][1] - tb[0][1])
        inter_area = comm.calc_overlap_area_of_two_rect(tb, box)
        iou_rat = inter_area / (t_area + box_area - inter_area)
        if iou_rat >= thresh_iou:
            return False
        contain_rat = inter_area / box_area
        if contain_rat >= thresh_contain:
            return False

    return True

class DetAndRecSizeElem:
    def __init__(self, raw_img, pageData, model, tat_det_model, inner_box):
        self.__img = raw_img
        self.__sub_imgs = dict()
        self.__boxes = list()
        self.__tat_boxes = list()

        self.__cut_img(raw_img, (720, 1280), (620, 1180))
        self.__predict(model)
        self.__nms()
        self.__predict_tat(raw_img, tat_det_model)
        self.__filter(pageData, inner_box)

    def __cut_img(self, img, size, step=None):
        H, W = img.shape[:2]
        h, w = size
        if step is None:
            step = (h // 2, w // 2)

        for y in range(0, H, step[0]):
            for x in range(0, W, step[1]):
                sub_img = img[y:min(y + h, H), x:min(x + w, W)]
                self.__sub_imgs[(y, x)] = sub_img

    def __predict(self, model):
        for offset, img in self.__sub_imgs.items():
            boxes, res, _ = model(img)

            rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            H, W = rot_img.shape[:2]
            rot_boxes, rot_res, _ = model(rot_img)
            for i in range(len(rot_boxes)):
                for point in rot_boxes[i]:
                    temp = point[0]
                    point[0] = point[1]
                    point[1] = W - temp
                rot_boxes[i] = [rot_boxes[i][3], rot_boxes[i][0], rot_boxes[i][1], rot_boxes[i][2]]

            boxes += rot_boxes
            res += rot_res

            for quad, info in zip(boxes, res):
                x0 = min(quad[0][0], quad[1][0], quad[2][0], quad[3][0])
                y0 = min(quad[0][1], quad[1][1], quad[2][1], quad[3][1])
                x1 = max(quad[0][0], quad[1][0], quad[2][0], quad[3][0])
                y1 = max(quad[0][1], quad[1][1], quad[2][1], quad[3][1])
                box = [[x0+offset[1], y0+offset[0]], [x1+offset[1], y1+offset[0]]]
                area = (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
                if area == 0:
                    continue
                box = comm.normalizeByImg(box, self.__img)
                self.__boxes.append({
                    'box': box,
                    'area': area,
                    'text': info[0],
                    'reserve': True
                })
                # print(f'---TEXT--- {info[0]}')

    def __nms(self, thresh=0.3):
        boxes = []
        self.__boxes = sorted(self.__boxes, key=lambda x: x['area'], reverse=True)
        for i in range(len(self.__boxes)):
            box = self.__boxes[i]
            if box['reserve']:
                boxes.append(box)
                for j in range(i + 1, len(self.__boxes)):
                    other = self.__boxes[j]
                    if other['reserve'] and (comm.calc_iou(box['box'], other['box']) >= thresh or
                                             comm.is_area_contain(box['box'], other['box'])):
                        self.__boxes[j]['reserve'] = False
        self.__boxes = boxes

    def __predict_tat(self, img, model):
        res = model(img)
        for box in res[0].boxes.xyxy.tolist():
            box = comm.normalizeByImg(box, img)
            self.__tat_boxes.append(comm.box_4x1_to_box_2x2(box))

    def __filter(self, pageData, inner_box):
        tmp_pageData = copy.deepcopy(pageData)
        for box in self.__boxes:
            if is_size_elem(box['box'], tmp_pageData, self.__tat_boxes, inner_box):
                pageData['targets'].append({
                    'box': box['box'],
                    'class': TargetTypeTag.sizeFlag,
                    'text': box['text']
                })
