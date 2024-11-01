import cv2
import numpy as np

img = cv2.imread(r'D:\Resources\Projects\drawing_rec\issue\PixPin_2024-08-31_10-16-13.png')
height = img.shape[0]
# 假设 img 是你的原始图像
# 转换成灰度图
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 边缘检测, Sobel算子大小为3
edges = cv2.Canny(image_gray, 170, 220, apertureSize=3)  # 170 220
# 霍夫曼直线检测，寻找像素大于0.6*height的直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=(0.3 * height), maxLineGap=20)

# 创建一个空白图像用于绘制直线
line_image = np.zeros_like(img)

# 遍历检测到的直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色线条，线宽为3

# 将绘制了直线的图像与原图合并
result = cv2.addWeighted(img, 0.8, line_image, 1, 0)

# 显示结果图像
cv2.imshow('Detected Lines', result)
cv2.waitKey(0)
cv2.destroyAllWindows()