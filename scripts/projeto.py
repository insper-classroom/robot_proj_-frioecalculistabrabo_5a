#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division
import rospy
import numpy as np
import numpy
import tf
import math
import cv2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped

from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sklearn.linear_model import LinearRegression

import visao_module

import cormodule

bridge = CvBridge()

cv_image = None
media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos


area = 0.0 # Variavel com a area do maior contorno

# Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
# Descarta imagens que chegam atrasadas demais
check_delay = False 

resultados = [] # Criacao de uma variavel global para guardar os resultados vistos

x = 0
y = 0
z = 0 
id = 0

frame = "camera_link"
# frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam

tfl = 0

tf_buffer = tf2_ros.Buffer()

centro2 = 0
media2 = 0

# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    print("frame")
    global cv_image
    global media
    global centro
    global resultados
    global area
    global centro2
    global media2

    now = rospy.get_rostime()
    imgtime = imagem.header.stamp
    lag = now-imgtime # calcula o lag
    delay = lag.nsecs
    # print("delay ", "{:.3f}".format(delay/1.0E9))
    if delay > atraso and check_delay==True:
        # Esta logica do delay so' precisa ser usada com robo real e rede wifi 
        # serve para descartar imagens antigas
        print("Descartando por causa do delay do frame:", delay)
        return 
    try:
        temp_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        mask = segmenta_linha_amarela(temp_image)
        contornos = encontrar_contornos(mask)
        cv2.drawContours(temp_image, contornos, -1, [0, 0, 255], 2)
        img, X, Y = encontrar_centro_dos_contornos(temp_image, contornos)
        img = desenhar_linha_entre_pontos(img, X,Y, (255,0,0))
        img, lm = regressao_por_centro(img, X,Y)
        angulo = calcular_angulo_com_vertical(img, lm)
        media, centro, maior_area, area =  cormodule.identifica_cor(img)
        img2 = temp_image.copy()
        media2, centro2, maior_area2, area2 =  cormodule.identifica_cor2(img2)
        cv2.imshow("Camera", temp_image) 
        cv2.waitKey(1)
        centro, saida_net, resultados =  visao_module.processa(temp_image)
        for r in resultados:
            # print(r) - print feito para documentar e entender
            # o resultado            
            pass

        # Desnecessário - Hough e MobileNet já abrem janelas
        cv_image = saida_net.copy()
        cv2.imshow("cv_image", cv_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        print('ex', e)

def segmenta_linha_amarela(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """
    img_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, (30, 55, 42), (32, 255, 255))
    final_mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((10, 10)))
    final_mask=morpho_limpa(final_mask)

    return final_mask

def morpho_limpa(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    mask = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel )
    mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, kernel )

    return mask

def encontrar_contornos(mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca os contornos encontrados
    """
    contornos, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
   
    return contornos

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)

def encontrar_centro_dos_contornos(img, contornos):
    """Não mude ou renomeie esta função
        deve receber um contorno e retornar, respectivamente, a imagem com uma cruz no centro de cada segmento e o centro dele. formato: img, x, y
    """
    X=[]
    Y=[]
    for contorno in contornos:
        M = cv2.moments(contorno)
        if M["m00"]:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            crosshair(img,(cX,cY),10,(0,0,255))    
            X.append(cX)
            Y.append(cY)

    return img, X, Y

def calcular_angulo_com_vertical(img, lm):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta.
        
        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """
    m, h = lm.coef_, lm.intercept_
    angulo=90+math.degrees(math.atan(m))

    return angulo

def desenhar_linha_entre_pontos(img, X, Y, color):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e retornar uma imagem com uma linha entre os centros EM SEQUENCIA do mais proximo.
    """
    for i in range(0,len(X)-1):
        cv2.line(img,(X[i],Y[i]),(X[i+1],Y[i+1]),color,5)
  
    return img    

def regressao_por_centro(img, x,y):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta e os parametros da reta
        
        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """

    x=np.array(x).reshape(-1,1)
    y=np.array(y).reshape(-1,1)
    lr_model=LinearRegression()
    lr_model.fit(x,y)

    m, h = lr_model.coef_, lr_model.intercept_

    x_min = int(min(x)) 
    x_max = int(max(x)) 

    y_min = int(m*x_min + h)
    y_max = int(m*x_max + h)    
    
    cv2.line(img, (x_min, y_min), (x_max, y_max), (255,255,0), thickness=3); 

    return img, lr_model

def checa_creeper():
    if not centro2 is None:
        return True
    else:
        False

if __name__=="__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/image/compressed"

    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)


    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

    tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
    tolerancia = 25

    AVANCAR = 0

    estado = AVANCAR

    ver_creeper = False
    
    try:
        # Inicializando - por default gira no sentido anti-horário
        vel = Twist(Vector3(0,0,0), Vector3(0,0,math.pi/10.0))
        
        while not rospy.is_shutdown():
            
            if estado == AVANCAR:
                if not ver_creeper:
                    vel = Twist(Vector3(0.5,0,0), Vector3(0,0,0))

                    if(len(centro) > 0 and len(media) > 0):
                        if (media[0] > centro[0]):
                            vel = Twist(Vector3(0.2,0,0), Vector3(0,0,-0.2))
                        else:
                            vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0.2))
                    else:
                        vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
                    ver_creeper = checa_creeper()
                else:
                    vel = Twist(Vector3(0,0,0), Vector3(0,0,0))



            for r in resultados:
                print(r)

            velocidade_saida.publish(vel)


            rospy.sleep(0.1)

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")


