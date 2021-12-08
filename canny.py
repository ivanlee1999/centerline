import cv2
import numpy as np
from numpy import polyfit, poly1d

import matplotlib.pyplot as plt

img = cv2.imread("test2.png")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img2 = cv2.equalizeHist(img2)
img2 = cv2.GaussianBlur(img2, (21, 21), 0)
#print(img2[:,100:150])


height, width = img2.shape

for w in range(0, width) :
    for h in range(0, height) :
        if(img2[h,w] >= 140):
            img2[h,w] = 140

filename = 'remove_spot.jpg'
cv2.imwrite(filename, img2)          

edges = cv2.Canny(img2, threshold1=20, threshold2=20)
print(edges.shape)



for w in range(0, 50) :
    for h in range(0, 50) :
        edges[h,w] =0

filename = 'edges_origin.jpg'
cv2.imwrite(filename, edges)

x1 = []
x2 = []
y1 = []
y2 = []

for w in range(0, width) :
    for h in range(0, height) :
        if(edges[h, w] == 255):
            if(y1 == []):
                x1.append(w)
                y1.append(h)
                edges[h,w] =0
                break
            elif(abs(h - y1[-1])<=15):
                x1.append(w)
                y1.append(h)
                edges[h,w] =0
                break

filename = 'edges.jpg'
cv2.imwrite(filename, edges)

p_upper = plt.plot(x1,y1)
plt.savefig('upper_edge.png')
plt.close()


for w in range(5, width) :
    for h in np.arange(height-1,-1,-1) :
        if(edges[h, w] == 255):
            if(h>140):
                x2.append(w)
                y2.append(h)
                break
            elif(abs(h - y2[-1])<=20):
                x2.append(w)
                y2.append(h)
                break

p_upper = plt.plot(x2,y2)
plt.savefig('lower_edge.png')
plt.close()


coeff1 = np.polyfit(x1, y1,15)
y1_fit = np.polyval(coeff1,range(0,324))

p_upper_fit = plt.plot(range(0,324),y1_fit)
p_upper = plt.plot(x1,y1)
plt.savefig('upper_edge_fit.png')



coeff2 = np.polyfit(x2, y2,15)
y2_fit = np.polyval(coeff2,range(0,324))

p_lower_fit = plt.plot(range(0,324),y2_fit)
p_lower = plt.plot(x2,y2)
plt.savefig('lower_edge_fit.png')
plt.close()

for i in range(0,324):
    if(y2_fit[i]>=153):
        y2_fit[i] =153

fit = np.zeros((154,324))
for w in range(0, width) :
    img2[int(y1_fit[w]),w] = 255
    img2[int(y2_fit[w]),w] = 255
    img2[int((y2_fit[w]+ y1_fit[w])/2),w] = 255

        

filename = 'fit.jpg'
cv2.imwrite(filename, img2)


