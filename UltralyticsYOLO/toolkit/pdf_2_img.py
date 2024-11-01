import os.path
import sys

import cv2
import numpy as np
from PIL import Image
from fitz import fitz
from tqdm import tqdm
import shutil

sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)  # 返回当前目录下的所有文件及文件夹的列表
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):  # 判断是否为文件
            os.remove(file_path)
        elif os.path.isdir(file_path):  # 判断是否为目录
            shutil.rmtree(file_path)  # 递归的删除文件


def clear_or_new_dir(dir):
    if os.path.exists(dir):
        del_file(dir)
    else:
        os.makedirs(dir)


def find_all_file_paths(base_path):
    res = []
    for root, _, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            res.append((file_path, str(file)))
    return res


def single_page_pdf_convert_to_img(pdf_path):
    imgs, rgb_imgs = [], []
    with fitz.open(pdf_path) as pdf:
        mat = fitz.Matrix(3, 3)
        i = 0
        for page in pdf:
            i += 1
            if i > 2:
                break
            pm = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
            rgb_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            imgs.append(img)
            rgb_imgs.append(rgb_image)
        return imgs, rgb_imgs


def http_single_page_pdf_convert_to_img(pdf_bytes, binary=True):
    pdf_document = fitz.open('pdf', pdf_bytes)

    page = pdf_document.load_page(0)
    mat = fitz.Matrix(3, 3)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    rgb_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    if binary:
        # 二值化处理
        thresh = 250
        _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    binary_img_3c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 灰度转 3 通道图片

    return binary_img_3c, rgb_image


def http_multi_page_pdf_convert_to_img(pdf_bytes):
    # 打开 PDF 文档
    pdf_document = fitz.open('pdf', pdf_bytes)

    # binary_images = []
    rgb_images = []

    # 遍历每一页
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        mat = fitz.Matrix(3, 3)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        rgb_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # if binary:
        #     # 二值化处理
        #     thresh = 250
        #     _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        # binary_img_3c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 灰度转 3 通道图片

        # # 添加到列表
        # binary_images.append(binary_img_3c)
        rgb_images.append(rgb_image)

    # return binary_images, rgb_images
    return rgb_images

def get_suffix(file_path):
    return os.path.splitext(file_path)[-1].lower()


def get_stem(file_path):
    return os.path.basename(file_path).split('.')[0]


# input_path = r'D:\Resources\Projects\qhx_ocr\input\test'
# output_path = r'D:\Resources\Projects\qhx_ocr\input\test\png'

if __name__ == "__main__":
    args = sys.argv[1:]
    input_path = args[0]
    output_path = args[1]

    clear_or_new_dir(output_path)

    i = 0
    for img_path, file_name in tqdm(find_all_file_paths(input_path)):
        i += 1
        if get_suffix(img_path).lower() == '.pdf':
            imgs = single_page_pdf_convert_to_img(img_path)[0]
            j = 0
            for img_ in imgs:
                _, img_ = cv2.threshold(img_, 250, 255, cv2.THRESH_BINARY)  # 二值化
                cv2.imwrite(os.path.join(output_path, get_stem(file_name) + f'[{j}]' + '.png'), img_)
                j += 1
