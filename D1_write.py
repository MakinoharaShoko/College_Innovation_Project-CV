import numpy as np
import cv2

img = cv2.imread('D1_res/watch.png',cv2.IMREAD_COLOR)


#基本的绘制操作
cv2.line(img,(0,0),(150,150),(255,255,255),2)#线
cv2.rectangle(img,(50,50),(150,150),(0,0,255),2)#矩形
cv2.circle(img,(100,100), 100, (0,255,0), 2)#圆


#自由绘制
pts = np.array([[0,0],[50,50],[100,150],[200,150]], np.int32)#坐标集
#pts = pts.reshape((-1,1,2))  可能会用到的操作
cv2.polylines(img, [pts], True, (0,255,255), 3)#绘制对象，坐标集，是否首尾相连，颜色，粗细



font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'TestText',(50,50), font, 1, (200,255,155), 2, cv2.LINE_AA)



cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()