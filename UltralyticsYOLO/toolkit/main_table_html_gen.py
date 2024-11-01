import os.path

from jinja2 import Environment, FileSystemLoader
import sys

sys.path.append('UltralyticsYOLO/toolkit')
import det_main_table_elem as dmte
import file_operations as fo
import comm_toolkit as comm
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, '../table_template')

# template_dir = r'D:\Resources\Projects\Python\ultralytics-main-2\UltralyticsYOLO\table_template'
# template_dir = r'/root/drawing_rec/ultralytics-main-2/UltralyticsYOLO/table_template'
# 创建一个 Jinja2 环境
env = Environment(loader=FileSystemLoader(template_dir))


def generate_html(template_path, output_path, context, isWrite=True, isOutputMap=False):
    template = env.get_template(template_path)

    # 渲染模板，未提供的变量默认为空字符串
    # html_content = template.render({k: context.get(k, '') for k in template.module.__dict__.keys()})
    html_content = template.render(context, items=context, isOutputMap=isOutputMap)

    if isWrite:
        # 将渲染后的内容写入新的 HTML 文件
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
    else:
        return html_content


template_file_paths = [
    r'comm_main_table_template.html',
    r'list_main_table_template.html',
    r'ap_list_main_table_template.html',
    r'ap_device_main_table_template.html',
    r'single_row_main_table_template.html'
]

valid_mainType = [0, 1, 2, 3, 4]


class MainTableHTMLGen:
    def __init__(self, img, pageData, html_output_dir, file_name, img_output_dir, isOutputMap=False):
        for t in pageData['targets']:
            if t['class'] in dmte.mainTableClsSet:
                # print(t)
                output_path = os.path.join(html_output_dir, fo.get_stem(file_name) + '.html')
                mainType = t['mainType']
                if mainType not in valid_mainType:
                    return
                generate_html(template_file_paths[mainType], output_path, t['text'], isOutputMap=isOutputMap)
                cv2.imwrite(os.path.join(img_output_dir, fo.get_stem(file_name) + '.png'), comm.extractImg(img, t['box']))


class HTTPMainTableHTMLGen:
    def __init__(self, pageData):
        self.html = None
        for t in pageData['targets']:
            if t['class'] in dmte.mainTableClsSet:
                if 'mainType' not in t:  # 未进行元素检测
                    return
                mainType = t['mainType']
                if mainType not in valid_mainType:
                    return
                self.html = generate_html(template_file_paths[mainType], '', t['text'], False, True)