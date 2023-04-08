import cv2
import os
import easyocr
import numpy as np
import time

min_width=80
min_height=80
draw_color=(0,255,0)

fps_limit=30
speed_unit='km/h'
pixel_to_meters_ratio=0.1
distance_between_lines=3


cap = cv2.VideoCapture(0)
frameWidth = 640
franeHeight = 480
plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
minArea = 500
reader = easyocr.Reader(['en'])
cap.set(3, frameWidth)
cap.set(4, franeHeight)
cap.set(10, 150)


fgbg = cv2.createBackgroundSubtractorMOG2()

old_center_list=[]
vehicle_speeds=[]


mp4_files = [filename for filename in os.listdir(os.getcwd()) if filename.endswith('.mp4')]
count_mp4_files = len(mp4_files)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'Video-{count_mp4_files}.mp4', fourcc, 20.0, (frameWidth, franeHeight))

count = 0

while True:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= min_width and h >= min_height:
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            center = (int(x + w / 2), int(y + h / 2))
            if len(old_center_list) > 0:
                old_center = old_center_list[-1]
                pixel_distance = np.sqrt((center[0] - old_center[0]) ** 2 + (center[1] - old_center[1]) ** 2)
                meters_distance = pixel_distance * pixel_to_meters_ratio
                seconds_elapsed = 1 / fps_limit
                speed = (meters_distance / seconds_elapsed) * 3.6
                vehicle_speeds.append(speed)
                if len(vehicle_speeds) > 1:
                    avg_speed = (vehicle_speeds[-1] + vehicle_speeds[-2]) / 2
                    cv2.putText(frame, f"{avg_speed:.2f} {speed_unit}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                draw_color, 2)
            old_center_list.append(center)

    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imgRoi = img[y:y + h, x:x + w]
            result = reader.readtext(imgRoi)
            if result:
                plateNumber = result[0][1]
                cv2.putText(img, plateNumber, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("ROI", imgRoi)

    cv2.imshow("Result", img)

    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        if result:
            cv2.imwrite(os.getcwd() + "\\" + str(count) + ".jpg", imgRoi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
