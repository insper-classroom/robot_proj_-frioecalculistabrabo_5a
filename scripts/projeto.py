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

centro2 = 0
media2 = 0

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
CHECK = False

x=None
y=None
global OK50
OK50 = False

angulo_robo=None
# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
	print("frame")
	global temp_image
	global media
	global centro
	global resultados
	global centro2
	global media2
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
		'''
		contornos = encontrar_contornos(mask)
		cv2.drawContours(temp_image, contornos, -1, [0, 0, 255], 2)
		img, X, Y = encontrar_centro_dos_contornos(temp_image, contornos)
		img = desenhar_linha_entre_pontos(img, X,Y, (255,0,0))
		img, lm = regressao_por_centro(img, X,Y)
		angulo = calcular_angulo_com_vertical(img, lm)
		'''
		media, centro, maior_area =  cormodule.identifica_cor(img)
		img2 = temp_image.copy()
		#media2, centro2, maior_area2 =  cormodule.identifica_cor2(img2)
		cv2.imshow("Camera", img) 
		cv2.waitKey(1)
		centro, saida_net, resultados =  visao_module.processa(temp_image)
		distance, ids =acha_aruco(temp_image)
		ids = ids[0][0]
		print("ids:",ids)
		cv2.imshow("Aruco",temp_image)
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

def multiplot(imgs, legenda="No sub"):
	""" Função """
	fig, axes = plt.subplots(1,len(imgs), figsize=(24,8))    
	fig.suptitle(legenda)
	if len(imgs)==1: # Peculiaridade do subplot. Não é relevante para a questão
		ax = axes
		ax.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
		return
	for i in range(len(imgs)):
		axes[i].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))

def multiplot_gray(imgs, legenda):
	""" Função que plota n imagens grayscale em linha"""
	fig, axes = plt.subplots(1,len(imgs), figsize=(26,8))    
	fig.suptitle(legenda)
	if len(imgs)==1: # Peculiaridade do subplot. Não é relevante para a questão
		ax = axes
		ax.imshow(imgs[0],  vmin=0, vmax=255, cmap="Greys_r")
		return
	for i in range(len(imgs)):
		axes[i].imshow(imgs[i], vmin=0, vmax=255, cmap="Greys_r")

def center_of_mass(mask):
	""" Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
	M = cv2.moments(mask)
	# Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	return [int(cX), int(cY)]

def ajuste_linear_x_fy(mask):
	"""Recebe uma imagem já limiarizada e faz um ajuste linear
		retorna coeficientes linear e angular da reta
		e equação é da forma
		y = coef_angular*x + coef_linear
	""" 
	pontos = np.where(mask==255)
	ximg = pontos[1]
	yimg = pontos[0]
	yimg_c = sm.add_constant(yimg)
	model = sm.OLS(ximg,yimg_c)
	results = model.fit()
	coef_angular = results.params[1] # Pegamos o beta 1
	coef_linear =  results.params[0] # Pegamso o beta 0
	return coef_angular, coef_linear


def ajuste_linear_grafico_x_fy(mask):
	"""Faz um ajuste linear e devolve uma imagem rgb com aquele ajuste desenhado sobre uma imagem"""
	coef_angular, coef_linear = ajuste_linear_x_fy(mask)
	print("x = {:3f}*y + {:3f}".format(coef_angular, coef_linear))
	pontos = np.where(mask==255) # esta linha é pesada e ficou redundante
	ximg = pontos[1]
	yimg = pontos[0]
	y_bounds = np.array([min(yimg), max(yimg)])
	x_bounds = coef_angular*y_bounds + coef_linear
	print("x bounds", x_bounds)
	print("y bounds", y_bounds)
	x_int = x_bounds.astype(dtype=np.int64)
	y_int = y_bounds.astype(dtype=np.int64)
	mask_rgb =  cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
	cv2.line(mask_rgb, (x_int[0], y_int[0]), (x_int[1], y_int[1]), color=(0,0,255), thickness=11);    
	return mask_rgb

def scaneou(dado):
    global distancia
    print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
    print("Leituras:")
    range = np.array(dado.ranges).round(decimals=2)
    distancia = range[0]
    print(range)
    #print("Intensities")
    #print(np.array(dado.intensities).round(decimals=2))

def acha_aruco(gray):
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




def angulo_q_roda(x,y, ang):
	angulo_trig = math.atan2(y,x)

	roda = (math.pi - ang) + angulo_trig

	return roda



def recebe_odometria(data):
	global x
	global y
	global contador
	global angulos
	global angulo_robo

	x = data.pose.pose.position.x
	y = data.pose.pose.position.y

	quat = data.pose.pose.orientation
	lista = [quat.x, quat.y, quat.z, quat.w]
	angulos = transformations.euler_from_quaternion(lista)  

	angulo_robo = angulos[2]

	#if contador % pula == 0:
		#print("Posicao (x,y)  ({:.2f} , {:.2f}) + angulo {:.2f}".format(x, y,angulos[2]))
	#contador = contador + 1

if __name__=="__main__":
	rospy.init_node("cor")

	topico_imagem = "/camera/image/compressed"

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
	VIU_50 = False
	
	try:
		# Inicializando - por default gira no sentido anti-horário
		vel = Twist(Vector3(0,0,0), Vector3(0,0,math.pi/10.0))
		
		while not rospy.is_shutdown():

			if angulo_robo is not None:
				if angulo_robo < 0:
					angulo_robo = angulo_robo + 2 * math.pi

			if AVANCAR:
				if not distance is None:
					VIU_50 = valida_aruco(50)
					if (distancia>1):
						vel = Twist(Vector3(0.3,0,0), Vector3(0,0,0))
	
						if(len(centro) > 0 and len(media) > 0):
							if (media[0] > centro[0]):
								vel = Twist(Vector3(0.2,0,0), Vector3(0,0,-0.2))
							else:
								vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0.2))
					if VIU_50:
						OK50 = True
						print("PASSEI!!!!!!!!!!", OK50)
					if (distancia<=1) and OK50:
						vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
						AVANCAR=False
						CREEPER = False
			else:
				vel = Twist(Vector3(0,0,0), Vector3(0,0,0.2))
				angulo_atual = angulo_robo 
				ang_roda = angulo_q_roda(x,y,angulo_atual)
				if angulo_robo+1 > ang_roda + angulo_atual:
					AVANCAR=True
					OK50 = False
					



			for r in resultados:
				print(r)

			velocidade_saida.publish(vel)


			rospy.sleep(0.1)

	except rospy.ROSInterruptException:
		print("Ocorreu uma exceção com o rospy")


