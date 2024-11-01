import json
import os.path


class ResultFileGen:
    def __init__(self, boxes, img_h: int, img_w: int, img_name):
        xyxy = boxes.xyxy.tolist()
        cls = boxes.cls.tolist()
        conf = boxes.conf.tolist()

        self.data = dict()
        self.data['imageHeight'] = img_h
        self.data['imageWidth'] = img_w
        self.data['imagePath'] = img_name
        self.data['targets'] = []
        for i in range(len(xyxy)):
            b = xyxy[i]
            quad = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
            self.data['targets'].append({
                'box': [[b[0], b[1]], [b[2], b[3]]],
                'quad': quad,
                'class': cls[i],
                'confidence': conf[i],
                'text': ''
            })

    def save(self, file_dir, file_name):
        with open(os.path.join(file_dir, file_name + '.json'), 'w', encoding='utf-8') as json_file:
            json.dump(self.data, json_file, indent=2)
