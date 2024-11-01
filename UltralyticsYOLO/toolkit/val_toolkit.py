import json

import numpy as np
import sys
sys.path.append('UltralyticsYOLO/toolkit')
import comm_toolkit as comm
from tag_data import TargetTypeTag


def calc_precision(tp, fp):
    if tp == 0:
        return 0
    return tp / (fp + tp)


def calc_recall(tp, fn):
    if tp == 0:
        return 0
    return tp / (fn + tp)


def stat_conf_matrix(conf_matrix, cls_name: list):
    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP

    res = dict()
    for i in range(len(TP)):
        res[str(i)] = {
            'name': cls_name[i],
            'TP（正确数）': TP[i],
            'FP（误判数）': FP[i],
            'FN（遗漏数）': FN[i],
            'Precision（精确率）': calc_precision(TP[i], FP[i]),
            'Recall（召回率）': calc_recall(TP[i], FN[i])
        }

    return res


def translate_output_res(name, data):
    return {
        'type（类型）': name,
        'TP（正确数）': data['TP'],
        'FP（误判数）': data['FP'],
        'FN（遗漏数）': data['FN'],
        'Precision（精确率）': calc_precision(data['TP'], data['FP']),
        'Recall（召回率）': calc_recall(data['TP'], data['FN'])
    }


def read_and_process_labelme_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    h = data['imageHeight']
    w = data['imageWidth']
    if 'shapes' in data:
        for t in data['shapes']:
            t['label'] = int(t['label'])
            t['points'] = comm.normalize(t['points'], h, w)

    if 'imageData' in data:
        del data['imageData']

    return data


def validate(truth_label, pred_label, iou_thresh=0.5, iot_thresh=0.8):
    res = dict()
    textRes = {'TP': 0, 'FP': 0, 'FN': 0}
    for tt in truth_label['shapes']:
        truth_cls = tt['label']
        max_iou = 0.0
        cand_pt = None
        for pt in [t for t in pred_label['pageData'][0]['targets'] if t['class'] == truth_cls]:
            iou = comm.calc_iou(tt['points'], pt['box'])
            iot = comm.calc_iot(tt['points'], pt['box'])
            if (iou >= iou_thresh or iot >= iot_thresh) and iou > max_iou:
                max_iou = iou
                cand_pt = pt

        if truth_cls not in res:
            res[truth_cls] = {'TP': 0, 'FP': 0, 'FN': 0}

        if cand_pt:
            cand_pt['TP'] = True
            res[truth_cls]['TP'] += 1
            if truth_cls in {TargetTypeTag.leadWire, TargetTypeTag.arrowAndFlag, TargetTypeTag.sectionalDrawingFlag}:
                if ('description' in tt) and len(tt['description']) > 0:
                    cand_pt['truthText'] = tt['description']
                    aggregate_text(textRes, validate_text(tt['description'], cand_pt['text']))
                else:
                    print(f'[!!!WARNING!!!] 存在实际文本为空的目标（cls={truth_cls}, box={tt["points"]}）')

        else:
            res[truth_cls]['FN'] += 1

    for pt in pred_label['pageData'][0]['targets']:
        if pt['class'] not in {TargetTypeTag.leadWire, TargetTypeTag.arrowAndFlag, TargetTypeTag.sectionalDrawingFlag, TargetTypeTag.sizeFlag}:
            continue

        if 'TP' not in pt:
            cls = pt['class']
            if cls not in res:
                res[cls] = {'TP': 0, 'FP': 0, 'FN': 0}
            res[cls]['FP'] += 1

    return res, textRes


def validate_text(truth_text, pred_text):
    res = {'TP': 0, 'FP': 0, 'FN': 0}
    stat_ed = set()
    for char in truth_text:
        if char not in stat_ed:
            res['TP'] += min(truth_text.count(char), pred_text.count(char))
            # res['FN'] += max(0, truth_text.count(char) - pred_text.count(char))
            stat_ed.add(char)

    res['FN'] = max(0, len(truth_text) - len(pred_text))
    res['FP'] = max(0, len(pred_text) - res['TP'])

    return res


def aggregate_box(total_data, frag_data):
    for k, v in frag_data.items():
        if k not in total_data:
            total_data[k] = {'TP': 0, 'FP': 0, 'FN': 0}
        for tk in total_data[k].keys():
            total_data[k][tk] += v[tk]


def aggregate_text(total_data, frag_data):
    for k in total_data.keys():
        total_data[k] += frag_data[k]

