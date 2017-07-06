from storage import db
import numpy as np
import cv2 as cv

ALPHA = 0.1
COLOR = (255,0,0)
RADIUS = 10

inputfile = "examples/road.png"
outfile = "examples/road-annotated.png"


output = cv.imread(inputfile)

for frame in db.iter_frames():
	if frame.height>150 or frame.width>300:
		continue
	x = int(frame.x)
	y = int(frame.y+frame.height/2)
	overlay = output.copy()
	overlay = cv.circle(overlay, (x, y), RADIUS, COLOR, -1)
	cv.addWeighted(overlay, ALPHA, output, 1 - ALPHA, 0, output)
	#x = int(frame.x)
	#y = int(frame.y)
	#w = int(frame.width)//2
	#h = int(frame.height)//2
	#cv.rectangle(output,(x-w,y-h),(x+w,y+h),(0,0,255),1)
	#cv.rectangle(output,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)


cv.imwrite(outfile,output)