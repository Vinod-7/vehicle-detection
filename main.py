import cv2
import numpy as np
from time import sleep

largura_min=80 
altura_min=80
#minimum width and height for the detected rectangle (vehicle).
offset=6 

pos_linha=550 
#setting a x-coordinate position for counting line
delay= 60 
#desired frame rate per second for the video playback.
detec = []
carros= 0

	
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
#capturing input video.
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()
#infinite loop for video processing.
while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    #Reading the next frame from the video and calculating the delay to control the frame rate.

    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #applying algorithm for subtracting the background and removing the noise.
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255,127,0), 3) 
    #drawing line for counting the vehicles.
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue
    #data annotation for detecting vehicles.
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(pos_linha+offset) and y>(pos_linha-offset):
                carros+=1
                cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0,127,255), 3)  
                detec.remove((x,y))
                print("vehicle is detected : "+str(carros))        
    #Calculating the center of the bounding rectangle for each contour.
    #adding it to the detection list, and checking if the center crosses the counting line.
    cv2.putText(frame1, "VEHICLE COUNT : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",dilatada)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
#Closing all OpenCV windows and releasing the video capture object