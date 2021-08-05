import cv2 as cv
import numpy as np
import math
import sys
import imutils
from os import listdir
from filter import filter_screws

'''
Does: list .png files
Arguments: images path (.png)
Returns: list of images names (without .png)
'''
def list_png_files(path=None):
    if path == None:
        print("Nenhuma pasta foi especificada.")
        return 0

    images = []
    files = [f for f in listdir(path)]
    for f in files:
        if f[len(f)-4:] == ".png":
            images.append(f)

    return images

def closing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Detecção de bordas
    gray = cv.Canny(gray,100,150)
    # Fechamento
    Kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))
    close = cv.morphologyEx(gray, cv.MORPH_CLOSE, Kernel)

    return close

def fill_holes(mask, size):
    mask = cv.bitwise_not(mask)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv.contourArea(contour) < size:
            cv.fillPoly(mask, pts =[contour], color=(0,0,0))

    mask = cv.bitwise_not(mask)

    return mask

'''
Does: find the two pcbs and crop them in to two different images
Arguments: image
'''
def segment_pcbs(image, screw_cascade):
    # resize
    if (image.shape[1] != 1920):
        image = imutils.resize(image, width=1920)

    # grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)  #<-- Precisso para o cascade

    #screws = screw_cascade.detectMultiScale(gray)
    screws = screw_cascade.detectMultiScale(gray, minNeighbors=3)

    # aqui a gente tem um erro quando não encontra nenhum screw, aparentemente não retorna um numpy array
    # ERRO: AttributeError: 'tuple' object has no attribute 'shape'
    try:
        if (screws.shape[0] < 4):
            pcbs = None
            return pcbs, pcbs
        elif screws.shape[0] >= 4:
            screws = filter_screws(gray, screws)
    except AttributeError:
        pcbs = None
        return pcbs, pcbs

    # Procurando o centro dos parafusos
    cx = 0.0
    cy = 0.0
    for key in screws:
        (x, y) = screws[key]
        cx = cx + x
        cy = cy + y
    pallet_center = (int(cx//4), int(cy//4))

    ang = np.degrees(np.arctan2(screws["left_bottom"][1] - screws["right_top"][1], screws["right_top"][0] - screws["left_bottom"][0]))
    # Ordem dos parafusos é: superior-esquerdo, superior-direito, inferior-esquerdo, inferior-direito

    ang = (ang - 2.61) + 180 # correção do ângulo dos parafusos
    ang *= -1

    rows, cols = image.shape[0], image.shape[1]
    cix = cols // 2
    ciy = rows // 2
    tx = cix - pallet_center[0]
    ty = ciy - pallet_center[1]
    # let M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, tx, 0, 1, ty])
    # M = np.array([[1, 0, tx], [0, 1, ty]])
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv.warpAffine(image, M, (cols, rows))
    image_center = (cix, ciy)

    # distância em pixels entre os screws
    c1 = np.sqrt((screws["left_bottom"][0] - screws["right_bottom"][0])**2 + (screws["left_bottom"][1] - screws["right_bottom"][1])**2)
    c2 = np.sqrt((screws["left_top"][0] - screws["right_top"][0])**2 + (screws["left_top"][1] - screws["right_top"][1])**2)
    c = (c1 + c2)/2

    M = cv.getRotationMatrix2D(image_center, ang, 1)
    image = cv.warpAffine(image, M, (cols, rows))

    #pxm = c/182  #pixels por milímetro
    pxm = c/163  #pixels por milímetro (distancia entre os parafusos)
    pcbx = 140 * pxm # largura da pcb em pixels (140 mm)
    pcby = 120 * pxm # altura da pcb em pixels
    extra = 25 * pxm # folga horizaontal

    lb = rows//2
    lt = int(lb - pcby)
    if lt < 0:
        lt = 0
    lr = cols//2
    ll = int(lr - pcbx)
    lr = int(lr + extra)
    if ll < 0:
        ll = 0

    #left = image[lt:lb, ll:lr,:].copy()
    left = image[lt:lb, ll:lr,:]

    rt = rows//2
    rb = int(rt + pcby)
    if rb > rows:
        rb = rows
    rl = cols//2
    rr = int(rl + pcbx)
    rl = int(rl - extra)
    if rr > cols:
        rr = cols

    #right = image[rt:rb, rl:rr, :].copy()
    right = image[rt:rb, rl:rr, :]
    return left, right
