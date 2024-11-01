import sys
import os
from PIL import ImageFont, ImageDraw, Image

sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm
import cv2
import numpy as np


palette = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 152, 255),
    (0, 255, 255)
]

current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, 'simfang.ttf')
font = ImageFont.truetype(font_path, 40)


def drawBoxes(img, data_dict):
    text = []
    for t in data_dict['pageData'][0]['targets']:
        cls = int(t['class'])
        absBox = comm.getAbsBoxByImg(t['box'], img)
        if cls <= 2 or cls == 5:
            img = cv2.rectangle(img, absBox[0], absBox[1], palette[cls], 2)
            # cv2.putText(img, t['text'], absBox[1], font, 1.0, (0, 0, 255), 2)
            text.append(((absBox[0][0], absBox[1][1]), t['text']))
        elif cls == 3:
            img = cv2.rectangle(img, absBox[0], absBox[1], palette[cls], 4)
            if 'cellBoxes' in t:
                for k, v in t['cellBoxes'].items():
                    absBox = comm.getAbsBoxByImg(v, img)
                    img = cv2.rectangle(img, absBox[0], absBox[1], (152, 0, 255), 2)
        elif cls == 4:
            img = cv2.rectangle(img, absBox[0], absBox[1], palette[cls], 4)
            if 'cellBoxes' in t:
                for boxes in t['cellBoxes']:
                    for k, v in boxes.items():
                        absBox = comm.getAbsBoxByImg(v, img)
                        img = cv2.rectangle(img, absBox[0], absBox[1], (152, 255, 0), 2)
        elif cls == 6:
            img = cv2.rectangle(img, absBox[0], absBox[1], palette[cls], 4)
            if 'cellBoxes' in t:
                for boxes in t['cellBoxes']:
                    for k, v in boxes.items():
                        absBox = comm.getAbsBoxByImg(v, img)
                        img = cv2.rectangle(img, absBox[0], absBox[1], (0, 152, 255), 2)

    img2 = Image.fromarray(img)
    drawImg = ImageDraw.Draw(img2)
    for pos, t in text:
        drawImg.text(pos, t, font=font, fill=(0, 0, 255))
    return np.array(img2)