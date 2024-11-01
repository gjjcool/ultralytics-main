import os.path

from jinja2 import Environment, FileSystemLoader
import sys

sys.path.append('UltralyticsYOLO/toolkit')
import det_main_table_elem as dmte
import file_operations as fo
import comm_toolkit as comm
import cv2
from tag_data import TargetTypeTag

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, '../table_template')
# template_dir = r'D:\Resources\Projects\Python\ultralytics-main-2\UltralyticsYOLO\table_template'
# template_dir = r'/root/drawing_rec/ultralytics-main-2/UltralyticsYOLO/table_template'
# 创建一个 Jinja2 环境
env = Environment(loader=FileSystemLoader(template_dir))


def generate_html(template_path, output_path, context, isWrite=True):
    template = env.get_template(template_path)

    # 渲染模板，未提供的变量默认为空字符串
    # html_content = template.render({k: context.get(k, '') for k in template.module.__dict__.keys()})
    # print(context)
    if (isinstance(context, list)) and (len(context) > 0) and (totalMass in context[0]):
        tm = context[0][totalMass]
    else:
        tm = ''

    html_content = template.render(items=context, order=tagMap, totalMass=tm)

    if isWrite:
        # 将渲染后的内容写入新的 HTML 文件
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
    else:
        return html_content


template_file_path = r'index_table_template.html'

tagMap = [
    'serialNumber',
    'deviceNo',
    'deviceName',
    'deviceNum',
    'material',
    'singlePiece',
    'total',
    'remark',
]

totalMass = 'totalMass'


class IndexTableHTMLGen:
    def __init__(self, img, pageData, html_output_dir, file_name, img_output_dir):
        i = 0
        for t in pageData['targets']:
            if t['class'] == TargetTypeTag.indexTable or t['class'] == TargetTypeTag.temp_apartDeviceIndexTable:
                i += 1
                output_path = os.path.join(html_output_dir, fo.get_stem(file_name) + f'[{i}].html')
                generate_html(template_file_path, output_path, t['text'])
                cv2.imwrite(os.path.join(img_output_dir, fo.get_stem(file_name) + f'[{i}].png'), comm.extractImg(img, t['box']))


class HTTPIndexTableHTMLGen:
    def __init__(self, pageData):
        self.htmlList = []
        for t in pageData['targets']:
            if t['class'] == TargetTypeTag.indexTable or t['class'] == TargetTypeTag.temp_apartDeviceIndexTable:
                html = generate_html(template_file_path, '', t['text'], False)
                self.htmlList.append(html)
