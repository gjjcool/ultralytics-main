import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from toolkit.predict import Predictor

app = FastAPI()
model = YOLO("runs/server/train13/weights/best.pt")  # YOLO 模型


def process_image(file_contents: bytes):
    # 将字节流转换为OpenCV图像
    nparr = np.frombuffer(file_contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_bytes(image):
    # 将OpenCV图像转换为字节流
    _, img_encoded = cv2.imencode('.png', image)
    return img_encoded.tobytes()


@app.post('/get_result_img')
async def get_result_img(file: UploadFile = UploadFile(...)):
    file_contents = await file.read()
    img = process_image(file_contents)
    p = Predictor(img)
    p.cut_img((720, 1280))
    p.predict(model)  # 预测
    p.nms()
    return StreamingResponse(io.BytesIO(cv2_to_bytes(p.draw_img())), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)

