import time
from redis import Redis
from celery.utils.log import get_task_logger
from app import create_celery_app
import gc

# Detection imports
import cv2 as cv
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from app.visao.preprocessing.segment_pcbs import *
from threading import Thread
from datetime import datetime
from gpiozero import Button
import argparse
import tflite_runtime.interpreter as tflite
from core.yolov4 import filter_boxes

celery = create_celery_app()
logger = get_task_logger(__name__)
r = Redis(host='all-in-one-redis', port=6379, db=0, decode_responses=True)

def makeDetection(frame, yolo1, yolo2, screw_cascade):
    
    # Inicializa array
    components = np.ones((6, 3))
        
    # np.array to list
    components = components.tolist()

    # Separa as duas PCBs
    pcb_left, pcb_right = segment_pcbs(frame, screw_cascade)

    if pcb_left is pcb_right:
        return pcb_left, pcb_left, components
    
    def draw_bboxes(index, image, bboxes):
        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = "{:.2f}".format(bbox[4])
            class_ind = int(bbox[5])
            correct = 1

            # Cores dos retangulos de cada classe
            rectangle_colors = {
                '3': (0, 0, 255), # Componente incorreto
                '0': (0, 255, 0), # Componente correto
            }

            # Cores dos scores de cada classe
            text_colors = {
                '3': (0, 0, 0),
                '0': (0, 0, 0),
            }

            placa = {
                'left': 0,
                'right': 1,
            }

            components_names = {
                '0': 'Azul',
                '1': 'Roxo 1',
                '2': 'Roxo 2',
                '3': 'Pequeno',
                '4': 'Preto',
                '5': 'Branco',
            }

            # Coordenadas do bounding box
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
            
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

    def detect1(index, image):
        image_data = cv.resize(image, (416, 416))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        input_details = yolo1.get_input_details()
        output_details = yolo1.get_output_details()

        yolo1.set_tensor(input_details[0]['index'], images_data)
        yolo1.invoke()

        predictions = [yolo1.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        bboxes_xyhw = np.asarray(predictions[0])
        scores = np.asarray(predictions[1])
        bboxes_xyhw = np.reshape(bboxes_xyhw, (bboxes_xyhw.shape[1], bboxes_xyhw.shape[2]))
        scores = np.reshape(scores, (scores.shape[1], scores.shape[2]))
        bboxes = filter_boxes(bboxes_xyhw, scores, score_threshold=0.25, input_shape=(416, 416))
        
        draw_bboxes(index, image, bboxes)

    def detect2(index, image):
        image_data = cv.resize(image, (416, 416))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        
        input_details = yolo2.get_input_details()
        output_details = yolo2.get_output_details()

        yolo2.set_tensor(input_details[0]['index'], images_data)
        yolo2.invoke()

        predictions = [yolo2.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        bboxes_xyhw = np.asarray(predictions[0])
        scores = np.asarray(predictions[1])
        bboxes_xyhw = np.reshape(bboxes_xyhw, (bboxes_xyhw.shape[1], bboxes_xyhw.shape[2]))
        scores = np.reshape(scores, (scores.shape[1], scores.shape[2]))
        bboxes = filter_boxes(bboxes_xyhw, scores, score_threshold=0.25, input_shape=(416, 416))
        
        draw_bboxes(index, image, bboxes)

    thread_1 = Thread(target=detect1,args=['left', pcb_left])

    thread_1.start()
    detect2('right', pcb_right)

    thread_1.join()

    def sort_func(x):
        return x[0]+x[1]

    components.sort(reverse=True, key=sort_func)

    return pcb_right,pcb_left,components

@celery.task(bind=True)
def long_task(self):
    # Inicialização
    step = 1
    components = {}
    self.update_state(state='INITIALIZING', meta={"step":step, "components":components})

    # Carregar a rede neural YOLO
    yolo1 = tflite.Interpreter('./tf-lite/yolov4-tiny-416.tflite')
    yolo1.allocate_tensors()
    yolo2 = tflite.Interpreter('./tf-lite/yolov4-tiny-416.tflite')
    yolo2.allocate_tensors()
    screw_cascade = cv.CascadeClassifier()
    screw_cascade.load(cv.samples.findFile("app/visao/screw_cascade.xml"))
    
    # Carregar pasta das imagens
    DEFAULT_MEDIA_FOLDER = os.environ.get("DEFAULT_MEDIA_FOLDER")

    butaoB = Button(2)
    butaoA = Button(3)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        step = 2
        self.update_state(state='READY FOR THE ACTION!', meta={"step":step, "components":components})

        # Pegar imagem da câmera
        ret, frame = cam.read()
        if not ret:
            break
        
        # Botão de saída
        if butaoA.is_pressed:
            step = 2
            self.update_state(state='WHY DID YOU LEFT ME?', meta={"step":step, "components":components})
            break

        # Detecção
        elif butaoB.is_pressed:
            step = 3
            self.update_state(state='DETECTION IN PROGRESS...', meta={"step":step, "components":components})
            date = datetime.now().strftime('%m_%d_%Y - %H:%S')
            cv.imwrite(DEFAULT_MEDIA_FOLDER+"_"+date+"_frame.png", frame)
            components.clear()
            pcbR,pcbL,components = makeDetection(frame, yolo1, yolo2, screw_cascade)
            if pcbR is pcbL:
                step = 5
                self.update_state(state="PCBS WERE NOT FOUND!", meta={"step":step, "components":components})
                time.sleep(5)
            else:
                cv.imwrite(DEFAULT_MEDIA_FOLDER+"left.jpg", pcbL)
                cv.imwrite(DEFAULT_MEDIA_FOLDER+"right.jpg", pcbR)
                step = 4
                self.update_state(state='SHOW TIME!', meta={"step":step, "components":components})
                gc.collect()
                time.sleep(10)
        else:
            # Inverter e escrever o frame na pasta
            #vframe = cv.flip(frame, -1)
            frame = cv.resize(frame, (640, 360))
            frame = cv.line(frame, (10, 180), (630, 180), (255, 0, ), 2)
            frame = cv.line(frame, (320, 10), (320, 350), (255, 0, ), 2)
            cv.imwrite(DEFAULT_MEDIA_FOLDER+"camera.jpg", frame)
            time.sleep(1) # Não esquentar tanto a raspi talvez

    return {'status': 'the task have been successfully processed'}
# start the task and send your ID to the frontend via Redis
task_id = r.get('taskid')
if not task_id:
    task = long_task.apply_async()
    r.set('taskid', task.id)
