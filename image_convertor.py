import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2



def crop(parameters, imagen):
    top_left_x = parameters['top_left_x']
    top_left_y = parameters['top_left_y']
    bottom_right_x = parameters['bottom_right_x']
    bottom_right_y = parameters['bottom_right_y']
    
    crop_height = bottom_right_y - top_left_y
    crop_width = bottom_right_x - top_left_x
    
    
    nueva_img = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    for i in range(top_left_y, bottom_right_y):
        for j in range(top_left_x, bottom_right_x):
            nueva_img[i - top_left_y, j - top_left_x] = imagen[i, j]
    crops = [crop_height, crop_width]
    
    return nueva_img

def invert_colors_and_binarize(image):
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binarizar la imagen: los píxeles por encima de 127 se convierten en 255 (blanco), los demás en 0 (negro)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # Invertir colores
    return binary_image

def rotate_image_fill(image, angle, fill_color=(255, 255, 255)):
    # Calculate the center of the image
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    
    # Get the rotation matrix for rotating the image around its center
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    # Calculate the size of the new image
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    
    bound_w = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
    bound_h = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)
    
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    # Perform the actual rotation and return the image
    rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h), borderValue=fill_color)
    return rotated_image

#Carga de placa recortada
placa_recortada = cv2.imread('nueva_img.jpg')


#Caracter 1
char1 = {
    'top_left_x':17,
    'top_left_y':40,
    'bottom_right_x':40,
    'bottom_right_y':74
}

char1_cropped = crop(char1, placa_recortada)
cv2.imshow('chars1 - Nueva_Img', char1_cropped)

#Caracter 2
char2 = {
    'top_left_x':38,
    'top_left_y':34,
    'bottom_right_x':52,
    'bottom_right_y':74
}

char2_cropped = crop(char2, placa_recortada)
cv2.imshow('chars2 - Nueva_Img', char2_cropped)

#Caracter 3
char3 = {
    'top_left_x':49,
    'top_left_y':32,
    'bottom_right_x':68,
    'bottom_right_y':64
}

char3_cropped = crop(char3, placa_recortada)
cv2.imshow('chars3 - Nueva_Img', char3_cropped)


#Caracter 4
char4 = {
    'top_left_x':75,
    'top_left_y':23,
    'bottom_right_x':91,
    'bottom_right_y':54
}

char4_cropped = crop(char4, placa_recortada)
cv2.imshow('chars4 - Nueva_Img', char4_cropped)

#Caracter 5
char5 = {
    'top_left_x':89,
    'top_left_y':16,
    'bottom_right_x':104,
    'bottom_right_y':51
}

char5_cropped = crop(char5, placa_recortada)
cv2.imshow('chars5 - Nueva_Img', char5_cropped)

#Caracter 6
char6 = {
    'top_left_x':101,
    'top_left_y':13,
    'bottom_right_x':117,
    'bottom_right_y':45
}

char6_cropped = crop(char6, placa_recortada)
cv2.imshow('chars6 - Nueva_Img', char6_cropped)

#Caracter 6
char7 = {
    'top_left_x':125,
    'top_left_y':3,
    'bottom_right_x':141,
    'bottom_right_y':40
}

char7_cropped = crop(char7, placa_recortada)
cv2.imshow('chars7 - Nueva_Img', char7_cropped)

# Aplicar la función a cada imagen recortada de los caracteres
char_images_cropped = [char1_cropped, char2_cropped, char3_cropped, char4_cropped, char5_cropped, char6_cropped, char7_cropped]
char_images_inverted = [invert_colors_and_binarize(img) for img in char_images_cropped]

# Visualizar una de las imágenes invertidas y binarizadas como ejemplo
cv2.imshow('Inverted and Binarized Character 1', char_images_inverted[0])
cv2.imwrite('imagen1.jpg', char_images_inverted[0])

char2_rotated = rotate_image_fill(char_images_inverted[1], -5, fill_color=(255, 255, 255))
cv2.imshow('Inverted and Binarized Character 2', char2_rotated)
cv2.imwrite('imagen2.jpg', char2_rotated) 
cv2.imshow('Inverted and Binarized Character 3', char_images_inverted[2])
cv2.imwrite('imagen3.jpg', char_images_inverted[2])
cv2.imshow('Inverted and Binarized Character 4', char_images_inverted[3])
cv2.imwrite('imagen4.jpg', char_images_inverted[3])
cv2.imshow('Inverted and Binarized Character 5', char_images_inverted[4])
cv2.imwrite('imagen5.jpg', char_images_inverted[4])
char6_rotated = rotate_image_fill(char_images_inverted[5], -25, fill_color=(255, 255, 255))
char7_rotated = rotate_image_fill(char_images_inverted[6], -10, fill_color=(255, 255, 255))

cv2.imshow('Inverted and Binarized Character 6', char6_rotated)
cv2.imwrite('imagen6.jpg', char6_rotated)

cv2.imshow('Inverted and Binarized Character 7', char7_rotated)
cv2.imwrite('imagen7.jpg', char7_rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.waitKey(0)

cv2.destroyAllWindows
