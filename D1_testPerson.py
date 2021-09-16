import numpy as np
import cv2

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy+ ih


def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

cap = cv2.VideoCapture('D2_res/55401403-1-208.mp4')

while (True):
    ret, frame = cap.read()  # ret代表是否有返回值，frame则是得到的帧
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度，没有转换前的就是frame

    #（可选）录制到视频并输出:
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # 分别显示灰度输出和彩色输出：
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    found, w = hog.detectMultiScale(frame)

    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q, in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
            else:
                found_filtered.append(r)
        for person in found_filtered:
            draw_person(frame, person)

    cv2.imshow("people detection", frame)

    #（可选）输出视频文件：
    #out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()