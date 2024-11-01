import json
import os.path

import UltralyticsYOLO.toolkit.file_operations as fo

def filter_field(data_dict):
    for page in data_dict['pageData']:
        for i in range(len(page['targets'])):
            target = page['targets'][i]
            if 'quad' in target:
                target.pop('quad')
            if 'confidence' in target:
                target.pop('confidence')

            # 去首尾空字符
            if 'text' in target:
                text = target['text']
                if isinstance(text, str):
                    target['text'] = text.strip()
                elif isinstance(text, list):  # 明细表/目录表
                    for i in range(len(text)):
                        for k, v in target['text'][i].items():
                            if isinstance(v, str):
                                target['text'][i][k] = v.strip()
                elif isinstance(text, dict):  # 主表
                    for k, v in target['text'].items():
                        if isinstance(v, str):
                            target['text'][k] = v.strip()

            # 去“明细表/目录表”中空行数据项
            if 'text' in target:
                text = target['text']
                if isinstance(text, list):  # 明细表/目录表
                    for i in range(len(text) - 1, -1, -1):
                        isEmptyRow = True
                        for k, v in target['text'][i].items():
                            if isinstance(v, str):
                                if len(v) > 0:
                                    isEmptyRow = False
                                    break
                        if isEmptyRow:
                            del target['text'][i]
                            if 'cellBoxes' in target:
                                del target['cellBoxes'][i]


json_path = r'D:\Resources\Projects\drawing_rec\input\test\0618\response_1717578929201.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    filter_field(data)

with open(os.path.join(fo.get_dir_name(json_path), 'new.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print('test')

