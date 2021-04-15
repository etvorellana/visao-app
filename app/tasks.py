import time
from celery.utils.log import get_task_logger
from app import create_celery_app

# Detection imports
import cv2 as cv
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
from app.visao.yolov3.utils import detect_image
from app.visao.yolov3.configs import *
from app.visao.yolov3.yolov4 import Create_Yolo
from app.visao.preprocessing.segment_pcbs import *
from threading import Thread
from datetime import datetime
from gpiozero import Button
import argparse

celery = create_celery_app()
logger = get_task_logger(__name__)

def makeDetection(frame, yolo, class_models):
    # Separa as duas PCBs
    pcb_left, pcb_right = segment_pcbs(frame)
    data = {}

    def draw_bboxes(index, image, bboxes):
        # Pegar bboxes azul_roxos
        azul_roxos = []
        for bbox in bboxes:
            class_ind = int(bbox[5])
            if class_ind in range(2):
                azul_roxos.append(bbox)

        # Ordenar azul_roxos por Y
        def takeY(elem):
            return elem[1]

        azul_roxos.sort(key=takeY)

        # Pegar resto dos bboxes
        for bbox in bboxes:
            class_ind = int(bbox[5])
            if class_ind in range(2, 5):
                azul_roxos.append(bbox)

        ind = 0
        for bbox in azul_roxos:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = "{:.2f}".format(bbox[4])
            class_ind = int(bbox[5])
            correct = 1

            # Cores dos retangulos de cada classe
            rectangle_colors = {
                '0': (0, 0, 255), # Componente incorreto
                '1': (0, 255, 0), # Componente correto
            }

            # Cores dos scores de cada classe
            text_colors = {
                '0': (0, 0, 0),
                '1': (0, 0, 0),
            }

            classes = {
                '0': 'azul',
                '1': 'roxo',
                '2': 'pequeno',
                '3': 'preto',
                '4': 'branco',
            }

            # Coordenadas do bounding box
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
            
            # Classificar se está correto ou incorreto
            prediction = np.array([[1.]])
            if class_ind in range(3):
                component = image[y1-5:y2+5,x1-5:x2+5,:]
                component = cv.resize(component, (32, 32))
                component = cv.cvtColor(component, cv.COLOR_BGR2GRAY)
                prediction = class_models[str(class_ind)].predict(component[np.newaxis,...,np.newaxis])

            # Verificar se azuis e roxos estão na ordem correta
            if ind == 0 and class_ind != 0:
                correct = 0

            if ind == 1 and class_ind != 1:
                correct = 0

            if ind == 2 and class_ind != 1:
                correct = 0
            ind += 1
            
            # Componente está incorreto
            if prediction[0][0]==0:
                correct = 0

            # Desenhar retângulo e score
            bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            cv.rectangle(image, (x1, y1), (x2, y2), rectangle_colors[str(correct)], bbox_thick)

            # get text size
            (text_width, text_height), baseline = cv.getTextSize(score, cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                                fontScale, thickness=1)
            # put filled text rectangle
            cv.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), 
                                            rectangle_colors[str(correct)], thickness=cv.FILLED)
            # put text above rectangle
            cv.putText(image, score, (x1, y1-4), cv.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, text_colors[str(correct)], 1, lineType=cv.LINE_AA)

            # colocar em data o indice da classe e se está certo ou errado
            data[index+"-"+classes[str(class_ind)]+"-"+str(ind)] = correct

    def detect(index, image):
        bboxes = detect_image(yolo, image, input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES,
                                score_threshold=0.5, iou_threshold=0.3)
        draw_bboxes(index, image, bboxes)

    thread_1 = Thread(target=detect,args=['left', pcb_left])

    thread_1.start()
    detect('right', pcb_right)

    thread_1.join()

    return pcb_right,pcb_left,data

@celery.task(bind=True)
def long_task(self):

    # Inicialização
    step = 1
    self.update_state(state='INITIALIZING', meta={"step":step})

    # Carregar a rede neural YOLO
    checkpoints_path = "app/visao/checkpoints/yolov3_C920-13all-50epochs_Tiny"
    yolo = Create_Yolo(input_size=416, CLASSES="app/visao/model_data/classes.txt")
    yolo.load_weights(checkpoints_path)

    # Carregar modelos de classificação 
    class_models = {
        '0': tf.keras.models.load_model('app/visao/classification_models/azul-roxo'),
        '1': tf.keras.models.load_model('app/visao/classification_models/azul-roxo'),
        '2': tf.keras.models.load_model('app/visao/classification_models/pequeno')
    }

    # step = 2
    # self.update_state(state='LOADING BUTTONS', meta={"step":step})

    butaoB = Button(2)
    butaoA = Button(3)

    # step = 3
    # self.update_state(state='LOADING CAMERA SETTINGS', meta={"step":step})

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        step = 2
        self.update_state(state='READY FOR THE ACTION!', meta={"step":step})

        # Pegar imagem da câmera
        ret, frame = cam.read()
        if not ret:
            break

        # Inverter e escrever o frame na pasta
        vframe = cv.flip(frame, -1)
        cv.imwrite("/usr/src/all-in-one/media/camera.jpg", vframe)
        time.sleep(1) # Não esquentar tanto a raspi talvez

        # Botão de saída
        if butaoA.is_pressed:
            step = 0
            self.update_state(state='WHY DID YOU LEFT ME?', meta={"step":step})

            break

        # Detecção
        elif butaoB.is_pressed:
            step = 3
            self.update_state(state='DETECTION IN PROGRESS...', meta={"step":step})

            pcbR, pcbL, data = makeDetection(frame, yolo, class_models)
            if pcbL.any() != None:
                cv.imwrite("/usr/src/all-in-one/media/left.jpg", pcbL)
            if pcbR.any() != None:
                cv.imwrite("/usr/src/all-in-one/media/right.jpg", pcbR)

            data["step"] = 4
            self.update_state(state='SHOW TIME!', meta={"data":data})
            time.sleep(15)

    return {'status': 'the task have been successfully processed'}