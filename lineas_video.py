import cv2
import numpy as np


#img = cv2.imread('exit_ramp.jpg')
cap = cv2.VideoCapture('solidWhiteRight.mp4')
#fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('D','I','V','X'), 30, (1280,720),True)
ret, frame = cap.read()

lim_inf_slope = 0.3
lim_sup_slope = 0.8

while (cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#gray = cv2.GaussianBlur(gray, (3,3), 0)
		edges = cv2.Canny(gray,50,150,apertureSize = 3)


		#cv2.imshow('gray',gray)

		rows, cols = edges.shape[:2] # dimension de la imagen

		#se definen vertices de la region de interes
		bottom_left  = [cols*0.17, rows*0.95]  #0.2 cols
		top_left     = [cols*0.5, rows*0.55]  #  0.55 rows
		bottom_right = [cols*0.95, rows*0.95]  
		top_right    = [cols*0.5, rows*0.55]  #0.5 cols  y 0.55 rows

		bottom_right_int = [cols*0.7, rows*0.95]
		top_right_int = [cols*0.5, rows*0.6]
		bottom_left_int = [cols*0.3, rows*0.95]
		top_left_int = [cols*0.5, rows*0.6]

		vertices = np.array([[bottom_left, top_left, top_right, bottom_right, bottom_right_int, top_right_int, top_left_int, bottom_left_int]], dtype=np.int32)
		mask = np.zeros_like(edges)
		#print ("mask", mask)

		if len(mask.shape)==2:
			poli = cv2.fillPoly(mask, vertices, 255)
			#print("si")
		else:
			print("no")
			cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel d
		#cv2.imshow('poligono', poli)

		select_region = cv2.bitwise_and(edges, mask)
		#cv2.imshow('region',select_region)

		minLineLength = 100  #30
		maxLineGap = 4   #10
		# se sacan lineas unicamente de la region de interes
		lines = cv2.HoughLinesP(select_region,1,np.pi/180,20,minLineLength,maxLineGap)

		left_lines = []
		left_weights = []
		right_lines = []
		right_weights = []
		contador_izq = 0
		contador_der = 0
		slope_izq = 0
		slope_der = 0
		inter_izq = 0
		inter_der = 0

		for x in range(0, len(lines)):
		    for x1,y1,x2,y2 in lines[x]:
			slope = float(y2 -y1) / (x2 -x1)
			if (x1 == x2 or y1 == y2):
				#print ('iguales')
				continue
			elif (slope > lim_inf_slope and slope < lim_sup_slope) or (slope < -lim_inf_slope and slope > -lim_sup_slope):
				#print('dif')
				cv2.line(gray,(x1,y1),(x2,y2),(0,0,0),2)
				intercept = y1 - slope*x1
				length = np.sqrt((y2-y1)**2+(x2-x1)**2)
				if slope < 0:
					contador_izq = contador_izq +1
					slope_izq = slope_izq + slope
					inter_izq = inter_izq + intercept 
					left_lines.append((slope, intercept))
					left_weights.append((length))
				else:
					contador_der = contador_der +1
					slope_der = slope_der+slope
					inter_der = inter_der + intercept
					right_lines.append((slope, intercept))
					right_weights.append((length))	

		#left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
		#right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
		slope_izq = slope_izq/contador_izq
		slope_der = slope_der/contador_der
		inter_izq = inter_izq/contador_izq
		inter_der = inter_der/contador_der

		#print('left',left_lane)
		#print('right',right_lane)

		y1 = gray.shape[0]
		y2 = 0.6*y1

		#print(left_lane[1])

		#x1 = int((y1 - left_lane[1])/left_lane[0])
		#x2 = int((y2 - left_lane[1])/left_lane[0])
		x1 = int((y1 - inter_izq)/slope_izq)
		x2 = int((y2 - inter_izq)/slope_izq)
		y1 = int (y1)
		y2 = int (y2)
		#print('coord izq',x1,y1,x2,y2)

		cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)

		y1 = gray.shape[0]
		y2 = 0.6*y1

		#print(right_lane[1])

		#x1 = int((y1 - right_lane[1])/right_lane[0])
		#x2 = int((y2 - right_lane[1])/right_lane[0])
		x1 = int((y1 - inter_der)/slope_der)
		x2 = int((y2 - inter_der)/slope_der)
		y1 = int (y1)
		y2 = int (y2)
		#print('coord der',x1,y1,x2,y2)


		cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
		#frame = cv2.flip(frame,180)
		out.write(frame)
		
		cv2.imshow('region',frame)

		if cv2.waitKey(1) & 0xFF == ord('c'):
			break
	else:
		break

cap.release()
out.release()
cv2.destroyAllWindows()

