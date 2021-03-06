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
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sklearn.linear_model import LinearRegression
from sensor_msgs.msg import LaserScan

import visao_module
import statsmodels.api as sm
import cormodule
import cv2.aruco as aruco
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


aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()
parameters.minDistanceToBorder = 0
parameters.adaptiveThreshWinSizeMax = 1000
marker_size  = 20
calib_path  = "/home/borg/catkin_ws/src/robot_proj_-frioecalculistabrabo_5a/"
camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_raspi.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_raspi.txt', delimiter=',')
font = cv2.FONT_HERSHEY_PLAIN

distance=20
ids = []
distancia=20
distancia_esq =20
coef_angular = 0
maior_area2 = 0
centro_creeper = []
centro_img = []


x=None
y=None


angulo_robo=None
# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    print("frame")
    global temp_image
    global media
    global centro
    global resultados
    global distance
    global ids

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
        img = ajuste_linear_grafico_x_fy(mask)
        """Aqui é onde definimos nossas missões, sendo estas a cor e o id"""
        #missao = ["orange", 11, "cow"]
        missao = ["blue", 12, "dog"]
        #missao = ["green", 23, "horse"]
        #missao = ["Teste", 0, 0]
        acha_creeper(missao, temp_image.copy())
        media, centro, maior_area =  cormodule.identifica_cor(mask)
        cv2.imshow("Camera", img) 
        cv2.waitKey(1)
        centro, saida_net, resultados =  visao_module.processa(temp_image)
        distance, ids =acha_aruco(temp_image)
        ids = ids[0][0]
        print("ids:",ids)
        cv2.imshow("Aruco", temp_image)
        cv2.waitKey(1)
        for r in resultados:
            # print(r) - print feito para documentar e entender
            # o resultado            
            pass

        # Desnecessário - Hough e MobileNet já abrem janelas
        #cv_image = saida_net.copy()
        #cv2.imshow("cv_image", img)
        #cv2.waitKey(1)
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


def ajuste_linear_x_fy(mask):
    global coef_angular
    """Recebe uma imagem já limiarizada e faz um ajuste linear
        retorna coeficientes linear e angular da reta
        e equação é da forma
        y = coef_angular*x + coef_linear
    """
    pontos = np.where(mask==255)
    ximg = pontos[1]
    yimg = pontos[0] 
    if len(yimg) != 0:
        yimg_c = sm.add_constant(yimg)
        model = sm.OLS(ximg,yimg_c)
        results = model.fit()
        coef_angular = results.params[1] # Pegamos o beta 1
        coef_linear =  results.params[0] # Pegamso o beta 0
        return coef_angular, coef_linear
    else:
        return 0,0


def ajuste_linear_grafico_x_fy(mask):
    """Faz um ajuste linear e devolve uma imagem rgb com aquele ajuste desenhado sobre uma imagem"""
    coef_angular, coef_linear = ajuste_linear_x_fy(mask)
    print("x = {:3f}*y + {:3f}".format(coef_angular, coef_linear))
    pontos = np.where(mask==255) # esta linha é pesada e ficou redundante
    ximg = pontos[1]
    yimg = pontos[0]
    if len(yimg) != 0:
        y_bounds = np.array([min(yimg), max(yimg)])
        x_bounds = coef_angular*y_bounds + coef_linear
        print("x bounds", x_bounds)
        print("y bounds", y_bounds)
        x_int = x_bounds.astype(dtype=np.int64)
        y_int = y_bounds.astype(dtype=np.int64)
        mask_rgb =  cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        cv2.line(mask_rgb, (x_int[0], y_int[0]), (x_int[1], y_int[1]), color=(0,0,255), thickness=11);    
        return mask_rgb
    else:
        return None

def scaneou(dado):

    #Função que analisa e detecta a distancia dos objetos na frente do robo pelo laser

    global distancia
    print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
    print("Leituras:")
    range = np.array(dado.ranges).round(decimals=2)
    distancia = range[0]
    distancia_e = range[359]
    distancia_d = range[1]
    distancia = (distancia + distancia_d + distancia_e)/3
    print("Distancia:",distancia)

def acha_aruco(gray):

    #Função cujo objetivo é identificar os ids dos arucos da pista

    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(ids)

    if ids is not None:
        #-- ret = [rvec, tvec, ?]
        #-- rvec = [[rvec_1], [rvec_2], ...] vetor de rotação
        #-- tvec = [[tvec_1], [tvec_2], ...] vetor de translação
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
        #-- Desenha um retanculo e exibe Id do marker encontrado
        aruco.drawDetectedMarkers(temp_image, corners, ids) 
        aruco.drawAxis(temp_image, camera_matrix, camera_distortion, rvec, tvec, 1)
        #-- Print tvec vetor de tanslação em x y z
        str_position = "Marker x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
        print(str_position)
        cv2.putText(temp_image, str_position, (0, 100), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        ##############----- Referencia dos Eixos------###########################
        # Linha referencia em X
        cv2.line(temp_image, (int(temp_image.shape[1]/2),int(temp_image.shape[0]/2)), ((int(temp_image.shape[1]/2) + 50),(int(temp_image.shape[0]/2))), (0,0,255), 5) 
        # Linha referencia em Y
        cv2.line(temp_image, (int(temp_image.shape[1]/2),int(temp_image.shape[0]/2)), ((int(temp_image.shape[1]/2)),(int(temp_image.shape[0]/2) + 50)), (0,255,0), 5) 	
        
        #####################---- Distancia Euclidiana ----#####################
        # Calcula a distancia usando apenas a matriz tvec, matriz de tanslação
        # Pode usar qualquer uma das duas formas
        distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
        distancenp = np.linalg.norm(tvec)
        #-- Print distance
        str_dist = "Dist aruco=%4.0f  dis.np=%4.0f"%(distance, distancenp)
        print(str_dist)
        cv2.putText(temp_image, str_dist, (0, 15), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

        return distance, ids
    return 1000,[[-1]]

def valida_aruco(target):
    if ids == target:
        return True
    else:
        return False

def recebe_odometria(data):

    #função cujo objetivo que fornece os angulos do robo frontal"

    global x
    global y
    global contador
    global angulos
    global angulo_robo

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    quat = data.pose.pose.orientation
    lista = [quat.x, quat.y, quat.z, quat.w]
    angulos = np.degrees(transformations.euler_from_quaternion(lista))  

    angulo_robo = angulos[2]
    angulo_robo = (angulo_robo + 360)%360

    #if contador % pula == 0:
        #print("Posicao (x,y)  ({:.2f} , {:.2f}) + angulo {:.2f}".format(x, y,angulos[2]))
    #contador = contador + 1

def acha_creeper(missao, frame):
    global centro_creeper
    global centro_img
    global maior_area2
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ####Escolhe cor
    if (missao[0] == "blue" and ids==missao[1]):  
        centro_creeper, centro_img, maior_area2 =  cormodule.identifica_cor_azul(frame)
        
    elif missao[0] == "green" and ids==missao[1]:
        centro_creeper, centro_img, maior_area2 =  cormodule.identifica_cor_verde(frame)

    elif missao[0] == "orange" and ids==missao[1]:
        centro_creeper, centro_img, maior_area2 =  cormodule.identifica_cor_laranja(frame)
    else:
        centro_creeper, centro_img, maior_area2 =  [0,0],[0,0],0
    ####
    #final_mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((10, 10)))
    #final_mask = morpho_limpa(final_mask)
    #centro_creeper = center_of_mass(final_mask)
    #centro_img = (frame.shape[1]//2, frame.shape[0]//2)
    cv2.imshow("creeper", frame)
    cv2.waitKey(1)



if __name__=="__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/image/compressed"
    ombro = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
    garra = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)
    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    recebe_odom = rospy.Subscriber("/odom", Odometry , recebe_odometria)
    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)

    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

    tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
    tolerancia = 25

    AVANCAR = True
    POSICIONAR = False
    PEGAR = False
    CREEPER = False
    RODANDO = False
    viu_creeper = False
    OK50 = False
    OK100 = False
    OK150 = False
    OK200 = False
    volta = False
    RODANDO_CREEPER=False
    recomeco = 1
    velocidade = 0.2

    velocidade_zero=Twist(Vector3(0,0,0), Vector3(0,0,0))
    
    
    try:
        # Inicializando - por default gira no sentido anti-horário
        vel = Twist(Vector3(0,0,0), Vector3(0,0,math.pi/10.0))
        
        while not rospy.is_shutdown():
            
            if angulo_robo is not None:
                if angulo_robo < 0:
                    angulo_robo = angulo_robo + 2 * math.pi

            if AVANCAR:
                print("AVANCAR!!!!!")
                print("Distancia:",distancia)
                if (coef_angular < 0.8) and (coef_angular > -0.8):
                    velocidade = 0.2
                else:
                    velocidade = 0.1 

                if not distance is None:
                    if 0.8 > coef_angular >- 0.8:
                        v = 0.2
                    else: 
                        v = 0.1
                    if (distancia>1):
                        vel = Twist(Vector3(velocidade,0,0), Vector3(0,0,0))
    
                        if(len(centro) > 0 and len(media) > 0):
                            if (media[0] > centro[0]):
                                vel = Twist(Vector3(velocidade,0,0), Vector3(0,0,-0.2))
                            else:
                                vel = Twist(Vector3(velocidade,0,0), Vector3(0,0,0.2))
                    if valida_aruco(50):
                        OK50 = True
                        OK100 = False
                        OK150 = False
                        OK200 = False

                    if valida_aruco(100):
                        OK100 = True
                        OK50 = False
                        OK150 = False
                        OK200 = False

                    if valida_aruco(150):
                        OK150 = True
                        OK50 = False
                        OK100 = False
                        OK200 = False

                    if valida_aruco(200):
                        OK200 = True
                        OK150 = False
                        OK100 = False
                        OK50 = False
                            
                    if (distancia<=1) and OK50:
                        vel = velocidade_zero
                        angulo_desejado = (angulo_robo - 180 + 360) % 360
                        print("GIRAR 50!")
                        AVANCAR = False 
                        RODANDO = True

                    if (distancia<=1.5) and OK200:
                        if not volta:
                            vel = velocidade_zero
                            angulo_desejado = (angulo_robo - 180 + 360) % 360
                            print("GIRAR 200!")
                            AVANCAR = False
                            volta = True
                            RODANDO = True

                    if (distancia<=1) and OK200 and volta:
                        vel = velocidade_zero
                        angulo_desejado = (angulo_robo - 315 + 360) % 360
                        print("GIRAR 200! VOLTA")
                        AVANCAR = False 
                        RODANDO = True 
                        
                    if (distancia<=1.5) and OK100:
                        if volta:
                            if not POSICIONAR:
                                agora = rospy.Time.now()
                                vel = velocidade_zero
                                POSICIONAR = True
                            if POSICIONAR:
                                vel = Twist(Vector3(0,0,0), Vector3(0,0,0.2))
                                if rospy.Time.now() - agora >= rospy.Duration.from_sec(0.5 * math.pi / 0.2):
                                    OK100 = False


                    if (distancia<=1) and OK150:
                        vel = velocidade_zero
                        angulo_desejado = (angulo_robo - 180 + 360) % 360
                        print("GIRAR 150!")
                        AVANCAR = False 
                        volta_esq = True 
                        RODANDO = True

                    if not centro_creeper is None and not viu_creeper:
                        if maior_area2 >= 1000:
                            AVANCAR = False
                            CREEPER = True
                            print("AREA CREEPER:",maior_area2)
            if RODANDO:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,0.2))
                print("ANG_ROBO:",angulo_robo,"ANG_DESEJADO:",angulo_desejado)
                if angulo_robo-5 < angulo_desejado < angulo_robo+5:
                    AVANCAR=True
                    RODANDO = False
                    OK50 = False
                    OK200 = False                

            if CREEPER:
                print("Distancia:",distancia)
                print("ENTROU CREEPER")
                ombro.publish(-0.4)
                if  distancia >= 0.2:
                    garra.publish(-1.0)
                    print("ENTROU CREEPER IF")
                    vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0))
                    if (centro_creeper[0] > centro_img[0]):
                        vel = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.1))
                    else:
                        vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0.1))
                elif distancia < 0.19:
                    vel = Twist(Vector3(0.0,0,0), Vector3(0,0,0))
                    garra.publish(0.0)
                    print("Distancia:",distancia)
                    print("ENTROU CREEPER ELSE")
                    CREEPER = False
                    viu_creeper = True
                    RODANDO_CREEPER = True
                    angulo_desejado = (angulo_robo - 180 + 360) % 360

            if RODANDO_CREEPER:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,0.2))
                ombro.publish(1.5)
                print("ANG_ROBO:",angulo_robo,"ANG_DESEJADO:",angulo_desejado)
                if angulo_robo-5 < angulo_desejado < angulo_robo+5:
                    AVANCAR=True
                    RODANDO_CREEPER = False
                    OK50 = False
                    OK200 = False 

    
            for r in resultados:
                print(r)

            velocidade_saida.publish(vel)


            rospy.sleep(0.1)

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")


