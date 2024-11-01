from ultralytics import YOLO


# from multiprocessing import freeze_support


def main():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="xxx.yaml", epochs=3000, imgsz=640, batch=-1, device=[1, 2, 3], iou=0.9)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == "__main__":
    # freeze_support()
    main()
