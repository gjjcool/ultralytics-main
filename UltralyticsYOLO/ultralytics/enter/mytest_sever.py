from ultralytics import YOLO
# from multiprocessing import freeze_support


def main():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="__rel_coco128.yaml", epochs=10)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    results = model(r"D:\Resources\Projects\Python\ultralytics-main-2\ultralytics-main\ultralytics\assets\bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == "__main__":
    # freeze_support()
    main()
