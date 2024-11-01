import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import numpy as np
import cv2
import json
import time
import copy
from PIL import Image, ImageDraw, ImageFont

import tools.infer.utility as utility
from tools.infer.predict_rec import TextRecognizer
from tools.infer.utility import get_rotate_crop_image


def get_model(dir):
    args = utility.parse_args()
    args.rec_model_dir = os.path.join(dir, "rec_infer")
    return TextRecognizer(args)


def rec_abc(cv2_image, pageData, text_recognizer):
    image_width = pageData['imageWidth']
    image_height = pageData['imageHeight']
    dt_boxes = []
    img_crop_list = []

    # 从字典中获取参数
    targets = pageData['targets']
    for target in targets:
        quad = [
            [coord[0] * image_width, coord[1] * image_height]
            for coord in target['quad']
        ]
        dt_boxes.append(quad)

    # 根据检测框裁剪图片
    for i in range(len(dt_boxes)):
        temp_box = copy.deepcopy(np.array(dt_boxes[i], dtype=np.float32))
        img_crop = get_rotate_crop_image(cv2_image, temp_box)
        img_crop_list.append(img_crop)

    # 识别裁剪后图片
    rec_res, elapse = text_recognizer(img_crop_list)

    # 把识别结果写入字典
    for index, target in enumerate(targets):
        target['text'] = rec_res[index][0]

    # # 把识别结果标在检测框右边
    # image = Image.fromarray(cv2_image)
    # draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype("./doc/simfang.ttf", 60)
    # for index, target in enumerate(targets):
    #     text_position = (target['box'][1][0] * image_width, target['box'][1][1] * image_height)
    #     text_color = (255, 0, 0)
    #     draw.text(text_position, target['text'], font=font, fill=text_color)
    # save_dir = './output/rec/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # image.save(save_dir + det_dict['imagePath'])



if __name__ == "__main__":
    args = utility.parse_args()
    args.rec_model_dir = "./model/rec_infer"
    text_recognizer = TextRecognizer(args)

    cv_image = cv2.imread("./doc/images/11-A.png")

    with open("./doc/jsons/11-A.json", 'r') as json_file:
        data_dict = json.load(json_file)

    rec_abc(cv_image, data_dict, text_recognizer)
