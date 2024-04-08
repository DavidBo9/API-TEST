import cv2
import numpy as np


imagen = cv2.imread('test.jpg')

top_left_x = 470
top_left_y = 600
bottom_right_x = 625
bottom_right_y = 689

crop_height = bottom_right_y - top_left_y
crop_width = bottom_right_x - top_left_x


nueva_img = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)


for i in range(top_left_y, bottom_right_y):
    for j in range(top_left_x, bottom_right_x):
        nueva_img[i - top_left_y, j - top_left_x] = imagen[i, j]


cv2.rectangle(imagen, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
cv2.imshow('Amongus - Original', imagen)
cv2.imshow('Amongus - Nueva_Img', nueva_img)
cv2.imwrite('nueva_img.jpg', nueva_img)

cv2.waitKey(0)
cv2.destroyAllWindows