import cv2


class PredictPipeline:
    def __init__(self, img_path, img_size=640):
        self.__img = cv2.imread(img_path)
        self.__sml_img = None
        self.__shrink_rate = 1
        self.target_size = None

        h, w = self.__img.shape[:2]
        if h >= w:
            self.__shrink_rate = h / img_size
            self.target_size = (int(w // self.__shrink_rate), int(img_size))
            self.__sml_img = cv2.resize(self.__img, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            self.__shrink_rate = w / img_size
            self.target_size = (int(img_size), int(h // self.__shrink_rate))
            self.__sml_img = cv2.resize(self.__img, self.target_size, interpolation=cv2.INTER_AREA)

    def get_tables_on_raw_img(self, bboxes, classes):
        main_table_imgs = []
        index_table_imgs = []

        for i in range(len(bboxes)):
            for j in range(4):
                bboxes[i][j] = int(bboxes[i][j] * self.__shrink_rate)

        for i in range(len(classes)):
            bb = bboxes[i]
            table_img = self.__img[bb[1]:bb[3], bb[0]:bb[2]]
            if int(classes[i]) == 0:
                main_table_imgs.append(table_img)
            elif int(classes[i]) == 1:
                index_table_imgs.append(table_img)

        return main_table_imgs, index_table_imgs

    def get_sml_img(self):
        return self.__sml_img
