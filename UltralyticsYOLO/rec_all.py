import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import cv2
import tools.infer.utility as utility
from PIL import Image, ImageDraw, ImageFont
from tools.infer.predict_system import TextSystem


def get_model():
    args = utility.parse_args()
    args.det_model_dir = "PaddleOCR/model/det_infer"
    args.cls_model_dir = "PaddleOCR/model/cls_infer"
    args.rec_model_dir = "PaddleOCR/model/rec_infer"
    args.use_angle_cls = True
    return TextSystem(args)


def get_sml_model():
    args = utility.parse_args()
    args.det_model_dir = "PaddleOCR/model/det_infer"
    args.cls_model_dir = "PaddleOCR/model/cls_infer"
    args.rec_model_dir = "PaddleOCR/model/rec_infer"
    args.det_db_unclip_ratio = 2.0
    return TextSystem(args)


def rec_all(cv2_image, text_system):
    height, width, channels = cv2_image.shape
    dt_boxes, rec_res, _ = text_system(cv2_image)
    rec_dict = []
    for index, box in enumerate(dt_boxes):
        quad = [[coord[0] / width, coord[1] / height] for coord in box]
        rec_dict.append({"quad": quad, "text": rec_res[index][0]})

    # 把识别结果标在检测框右边
    # image = Image.fromarray(cv2_image)
    # draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype("./doc/simfang.ttf", 60)
    # for index, target in enumerate(rec_dict):
    #     text_position = (target['quad'][3][0] * width, target['quad'][3][1] * height)
    #     text_color = (255, 0, 0)
    #     draw.text(text_position, target['text'], font=font, fill=text_color)
    # save_dir = './output/rec_all/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # image.save(save_dir + "11.png")

    return rec_dict


if __name__ == "__main__":
    args = utility.parse_args()
    args.det_model_dir = "./model/det_infer"
    args.cls_model_dir = "./model/cls_infer"
    args.rec_model_dir = "./model/rec_infer"

    text_system = TextSystem(args)

    cv_image = cv2.imread("./doc/images/11-B.png")
    rec_all(cv_image, text_system)
