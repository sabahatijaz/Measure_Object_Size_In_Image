import PySimpleGUI as sg
sg.theme('Light Blue 2')

layout = [[sg.Text('Enter Fundus Image file ')],
          [sg.Text('File ', size=(8, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Submit(), sg.Cancel()]]

window = sg.Window('Image ', layout)

event, values = window.read()
window.close()
#print(f'You clicked {event}')
#print(f'You chose filenames {values[0]}')

text_input = values[0]
#realimg = cv2.imread(text_input)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def find_shape(approx):
    x, y, w, h = cv2.boundingRect(approx)
    if len(approx) == 3:
        s = "Triangle"
    elif len(approx) == 4:
        calculation = w / float(h)
        if calculation >= 0.95:
            s = "Square"
        else:
            s = "Rectangle"
    elif len(approx) == 5:
        s = "Pentagon"
    elif len(approx) == 8:
        s = "Octagon"
    else:
        s = "Circle"
    return s, x, y, w, h

image = cv2.imread(text_input) 
low=np.array([0,0,0])
high=np.array([200,200,200])
imm=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(imm,low,high)
ime=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow("hjgjh",mask)
#imgplot = plt.imshow(mask)
#plt.show()
img2_fg = cv2.bitwise_and(mask,ime)
#imgplot = plt.imshow(img2_fg)
#plt.show()
img_contours, hierarchy = cv2.findContours(img2_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image, img_contours, -1, (0, 255, 0))
#cv2.imshow('Image Contours', image)
BLACK_THRESHOLD = 200
THIN_THRESHOLD = 100
idx = 0

#root_ext = os.path.splitext(filename) 
#path = os.path.join('AWS_IMG/', root_ext[0]) 
for cnt in img_contours:
    area = cv2.contourArea(cnt)
    if area > 9000:
        idx += 1
        # Find length of contours
        param = cv2.arcLength(cnt, True)
        # Approximate what type of shape this is
        approx = cv2.approxPolyDP(cnt, 0.01 * param, True)
        #cv2.drawContours(imm, cnt, -1, (255, 255, 0), 10)
        x, y, w, h = cv2.boundingRect(cnt)
        shape, x, y, w, h = find_shape(approx)
        #shape, x, y, w, h = find_shape(approx)
        #roi = image[y:y + h, x:x + w]
        #x, y, w, h = cv2.boundingRect(cnt)
        #epsilon = 0.1*cv2.arcLength(cnt,True)
        #approx = cv2.approxPolyDP(cnt,epsilon,True)
        #roi = ime[y:y + h, x:x + w]
        #if h < THIN_THRESHOLD or w < THIN_THRESHOLD:
            #continue
        #path = 'Documents/AWS_IMG/'
        #cv2.imwrite(("trial10"+str(idx) + '.jpg'), roi)
        #cv2.imwrite(filename+str(idx) + '.jpg', roi)
        #cv2.rectangle(ime, (x, y), (x + w, y + h), (200, 0, 0), 2)
#print(dist)
# Find the index of the largest contour
# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in img_contours]
max_index = np.argmax(areas)
cnt=img_contours[max_index]
#idx += 1
    # Find length of contours
param = cv2.arcLength(cnt, True)
#print(param)
# Approximate what type of shape this is
approx = cv2.approxPolyDP(cnt, 0.01 * param, True)
#cv2.drawContours(imm, cnt, -1, (255, 255, 0), 10)
x, y, w, h = cv2.boundingRect(approx)
shape, x, y, w, h = find_shape(approx)
#shape, x, y, w, h = find_shape(cnt)
print("height: ",h)
print("width: ",w)
print("length1: ",x)
print("length2: ",y)
import PySimpleGUI as sg
import PySimpleGUI as sg
import io
from PIL import Image
sg.theme('Light Blue 2')
layout = [
    [sg.Output(key='-OUT-', size=(50, 10))],
        [sg.Image(key="-IMAGE-")],
]
window = sg.Window("Image Viewer", layout,finalize=True,auto_close=True)
#window = sg.Window('Image shape Analysis', layout, element_justification='center').finalize()
window['-OUT-'].TKOut.output.config(wrap='word') # set Output element word wrapping

#print(dist)
print("height: ",h)
print("width: ",w)
print("length1: ",x)
print("length2: ",y)
#print("shape: ",shape)
image = Image.open(text_input)
image.thumbnail((800, 800))
bio = io.BytesIO()
image.save(bio, format="PNG")
window["-IMAGE-"].update(data=bio.getvalue())
while True:
    win, ev, val = sg.read_all_windows()
    if ev == sg.WIN_CLOSED:
        win.close()
        break
window.close()