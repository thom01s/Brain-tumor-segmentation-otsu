# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:44:19 2022

@author: Thomas
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#leitura da imagem
img = cv2.imread('meningioma/Tr-me_0024.jpg',0)
#img = cv2.imread('notumor/Tr-no_1575.jpg',0)

#hough circles
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,150,param1=127,param2=50,minRadius=100,maxRadius=170)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # círculo de fora
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,0,255),2)
    # centro do círculo
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

#máscara    
mask = np.full((cimg.shape[0], cimg.shape[1]),0,dtype=np.uint8)
for j in circles[0, :]: cv2.circle(mask, (j[0], j[1]), j[2], (255, 255, 255), -1)
mask = cv2.bitwise_and(img,img,mask = mask)


#filtro gaussiano
blur = cv2.GaussianBlur(mask,(5,5),0)

equal = cv2.equalizeHist(blur)

#retirada de pontos pouco intensos por otsu e equalização do histograma
ret, high = cv2.threshold(equal,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
equal = cv2.equalizeHist(high)
#repetindo
ret, high = cv2.threshold(equal,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
equal = cv2.equalizeHist(high)

#binarização em 50% da intensidade total
ret, binary = cv2.threshold(equal,85,255,cv2.THRESH_BINARY)
#cv2.imshow("binario", binary)


contagem = np.sum(binary == 255)
  
#contornos
kernel = np.ones((3,3),np.uint8)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#operações morfológicas
count = 0
while(len(contours) > 1):  
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations = count)
    contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    dilation = cv2.dilate(opening, kernel, iterations = count)
    count +=1
skull = binary - dilation
ret, skull = cv2.threshold(skull,127,255,cv2.THRESH_BINARY)
tumor = binary - skull

#coloração do tumor
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)         
for i in range(0,tumor.shape[0]):
    for j in range(0,tumor.shape[1]):
        if tumor[i,j] == 255:
            color[i,j] = [0,0, 255]

#medção de tamanho
area = np.sum (tumor == 255)

#plot para o display
cv2.imshow("tumor", color)

plt.subplot(331),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(cimg, cmap='gray'),plt.title('Circles')
plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(mask, cmap='gray'),plt.title('Retirada da parte externa')
plt.xticks([]), plt.yticks([])
plt.subplot(334),plt.hist(img.ravel(),256),plt.title('Histograma inicial')
plt.xticks([0, 50, 100, 150, 200, 255]), plt.yticks([])
plt.subplot(335),plt.hist(equal.ravel(),256),plt.title('Histograma equalizado')
plt.xticks([0, 50, 100, 150, 200, 255]), plt.yticks([])
plt.subplot(336),plt.imshow(equal, cmap='gray'),plt.title('equalizado')
plt.xticks([]), plt.yticks([])
plt.subplot(337),plt.imshow(binary, cmap='gray'),plt.title('Binarização')
plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(dilation, cmap='gray'),plt.title('Massa')
plt.xticks([]), plt.yticks([])
plt.subplot(339),plt.imshow(skull, cmap='gray'),plt.title('Crânio')
plt.xticks([]), plt.yticks([])
plt.show()