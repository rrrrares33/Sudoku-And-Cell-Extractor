import os
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

# =============================================================================================================================

sharpening_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]])

def apply_filter_to_images(images, to_gray, blur_level, sharpen, treshold):
    """
        Applies some filters and prelucrations against the provided raw images.

    Args:
        images ([list()]): the list of raw images.
        to_gray ([bool]): true to convert images to gray.
        blur_level ([integer]): an odd number bigger then 3 representing the level of blur
            that should be added to the image.
        sharpen ([integer]): an odd number bigger then 3 representing the level of sharpening
            that should be added to the image.
        treshold ([String]): the type of treshold that should be use. 'adaptive' or 'otsu'

    Returns:
        [filtered_images]: List of images after all filters were applied.
    """
    return_images = list()
    for image in images:
        read_color = cv2.IMREAD_GRAYSCALE
        aux_image = image
        if to_gray:
            aux_image = cv2.cvtColor(aux_image, cv2.COLOR_BGR2GRAY)

        if sharpen: 
            aux_image = cv2.filter2D(src=aux_image, ddepth = -1, kernel = sharpening_kernel)

        # aux_image = cv2.GaussianBlur(aux_image, (blur_level, blur_level), 0)

        if treshold == 'adaptive':
            aux_image = cv2.adaptiveThreshold(aux_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7)
        elif treshold == 'otsu':
            aux, aux_image = cv2.threshold(aux_image, 10, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        return_images.append(aux_image)

    return return_images

# =============================================================================================================================

def sortFunc(e):
    return e[0]

def get_biggest_square_corners(image):
    """
    Gets the biggest form in the image, in our case, all of the time it will be the square margin.

    Args:
        image ([list]):

    Returns:
        listCornerPoints[list]: A list of points(x,y) that appear on the biggest square in the image.
    """
    squares, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_pos = 0
    biggest_square = squares[0]
    for i in range(len(squares)):
        if cv2.contourArea(squares[i]) > biggest_pos:
            biggest_square = squares[i]
            biggest_pos = cv2.contourArea(squares[i])
    
    rectang = cv2.minAreaRect(biggest_square)
    cornerPoints = cv2.boxPoints(rectang)
    cornerPoints = np.int0(cornerPoints)

    listCornerPoints = list()
    for aux in cornerPoints:
        listCornerPoints.append((aux[0],aux[1]))
    
    listCornerPoints.sort(key = sortFunc)

    return listCornerPoints 

# =============================================================================================================================

def rotate_the_images_to_upright(images):
    """
    In case the upper-edges of the square are not in-line, it rotates the image until they are in line.
    This is done in order to prevent highly tilted images.

    Args:
        images ([list])

    Returns:
        upright_images[list]: Images tilted upwards.
    """
    upright_images = list()
    for image in images:
        corners = get_biggest_square_corners(image = image)
        difference = corners[1][0] - corners[0][0]
        if difference < 20:
            pass
        else:
            aux_image1 = imutils.rotate(image, -0.8)
            aux_image2 = imutils.rotate(image, 0.8)
            corners1 = get_biggest_square_corners(image = aux_image1)
            corners2 = get_biggest_square_corners(image = aux_image2)
            difference1 = corners1[1][0] - corners1[0][0]
            difference2 = corners2[1][0] - corners2[0][0]

            rotationDirection = 0
            if difference1 < difference2:
                rotationDirection = -0.8
            else:
                rotationDirection = 0.8


            while(difference > 20):
                image = imutils.rotate(image, rotationDirection)
                corners = get_biggest_square_corners(image = image)
                difference = corners[1][0] - corners[0][0]
        
        upright_images.append(image)

    return upright_images

# =============================================================================================================================

def crop_the_images(images, offset = 0):
    """Crops the biggest square in the image using its corners.

    Args:
        images (list): list of images
        offset (int, optional): if added, it adds a little padding to the cutting area. Defaults to 0.

    Returns:
        cropped_images(list): the results of the cropping.
    """
    cropped_images = list()
    for image in images:
        corners = get_biggest_square_corners(image = image)
        # print(corners)
        x_loc = min(corners[0][0], corners[1][0])
        y_loc = min(corners[0][1], corners[1][1])
        width = max(corners[2][0], corners[3][0]) - x_loc
        height = max(corners[2][1], corners[3][1]) - y_loc
        #print(x_loc, y_loc, width, height)
        cropped_images.append(image[y_loc + offset: y_loc + height - offset, x_loc + offset: x_loc + width - offset])

    return cropped_images

# =============================================================================================================================

def break_the_images_in_81_tiles(images, offset = 0):
    """
    Breaks each sudoku image in 81 smaller images, each one representing a square from the sudoku big square.
    This is done pretty simple, assuming that the cropping of the main square was done right, we can crope
        each little square by considering that each tile occupies exactly width//9 and height//9 in the sudoku.
    Offset is there in order to prevent margins from appear inside of the cropped square (for easier number detection)

    Args:
        images (list): cropped images of sudoku.
        offset (int, optional): [description]. Defaults to 0.

    Returns:
        breaked_images(list): the images breaked into 81 tiles.
    """
    breaked_images = list()

    for image in images:
        image_tiles = list()
        width, height = image.shape[1] , image.shape[0]
        for line in range(9):
            curr_line = list()
            for col in range(9):
                curr_line.append(image[height//9 * line + offset : height//9 * (line+1) - offset, width//9 * col + offset: width//9 * (col+1) - offset])
            image_tiles.append(curr_line)

        breaked_images.append(image_tiles)

    return breaked_images

# ================================================================================================================================

def get_structure_matrix(breaked_down_tiles, threshold = 500):
    """
    It gets an list with the 81 tiles of an image.
    It should return a matrix of (9, 9), consisting of 'x' and 'o'.
    'o' represents that the sum of white pixels from the small square 
        from its position was smaller or equal to 500. 
    'x' represents that the sum of white pixels from the small square 
        from its position was bigger then 500. 

    Args:
        breaked_down_tiles ([type]): a list containing the tiles of an image.

    Returns:
        return_matrix(list(list)): structure matrix of the sudoku
    """
    structure_matrix = list()
    for each_line_of_tiles in breaked_down_tiles:
        for each_tile in each_line_of_tiles:
            nr_of_white_pixels = 0
            for line_of_pixels in each_tile:
                for pixel in line_of_pixels:
                    if pixel > 100:
                        nr_of_white_pixels += 1
            if nr_of_white_pixels > threshold:
                structure_matrix.append('x')
            else:
                structure_matrix.append('o')
    return_matrix = list()
    aux = list()
    for x in structure_matrix:
        aux.append(x)
        if len(aux) == 3:
            return_matrix.append(aux)
            aux = list()
    return return_matrix