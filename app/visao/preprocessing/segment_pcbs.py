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
    screws = screw_cascade.detectMultiScale(gray, minNeighbors = 5)
    cx = 0.0
    cy = 0.0

    # aqui a gente tem um erro quando não encontra nenhum screw, aparentemente não retorna um numpy array
    # ERRO: AttributeError: 'tuple' object has no attribute 'shape'
    try: 
        if (screws.shape[0] < 4):
            pcbs = None
            return pcbs, pcbs
        elif screws.shape[0] >= 4:
            screws = filter_screws(image, screws)
    except AttributeError:
        pcbs = None
        return pcbs, pcbs

    
    # Procurand o centro de rotação
    for(x, y) in screws:
        cx = cx + x
        cy = cy + y
    center = (int(cx//4), int(cy//4))
    
    # Ordenando os screws para dimensionar a imagem
    scrD = {}
    for(x, y) in screws:
        if (x < center[0]):
            if (y < center[1]):
                scrD["lt"] = (x, y)
            else:
                scrD["lb"] = (x, y)
        else: 
            if (y < center[1]):
                scrD["rt"] = (x, y)
            else:
                scrD["rb"] = (x, y)

    # equalização adaptativa
    #clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    #gray = clahe.apply(gray[:100,600:1200])

    # detecção de bordas
    #gray = cv.Canny(gray,100,150)

    # encontrar maior linha da imagem (esperamos que seja sempre a horizontal da placa)
    #lines = cv.HoughLines(gray,1,np.pi/180,150)
    #try:
    #    if lines == None:
            #image_rotated = imutils.rotate_bound(image, -45)
    #        ang = -45
    #except ValueError:
    #    rho, theta = lines[0][0][0], lines[0][0][1]
        #image_rotated = imutils.rotate_bound(image, theta*57.2-45)
        #ang = np.degrees(theta) - 45
        ang = np.degrees(np.arctan2(scrD["lb"][1] - scrD["rt"][1], scrD["rt"][0] - scrD["lb"][0]))
        
    rows,cols = image.shape[0], image.shape[1]
    # distância em pixels entre os screws
    c1 = np.sqrt((scrD["lb"][0] - scrD["rb"][0])**2 + (scrD["lb"][1] - scrD["rb"][1])**2)
    c2 = np.sqrt((scrD["lt"][0] - scrD["rt"][0])**2 + (scrD["lt"][1] - scrD["rt"][1])**2)
    c = (c1 + c2)/2
    ang *= -1
    
    M = cv.getRotationMatrix2D(center,ang,1)
    image = cv.warpAffine(image,M, (cols,rows))
    
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