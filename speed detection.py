import cv2
import numpy as np
import os

min_width=20
min_height=20
draw_color=(0,255,0)

fps_limit=30
speed_unit='km/h'
pixel_to_meters_ratio=0.1
distance_between_lines=3

cap = cv2.VideoCapture(0)

car_cascade = cv2.CascadeClassifier(f"{os.getcwd()}/xml version=4.0.txt")

fgbg = cv2.createBackgroundSubtractorMOG2()

old_center_list=[]
vehicle_speeds=[]

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgmask = fgbg.apply(gray)

        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w>=min_width and h>=min_height:
                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                center=(int(x+w/2),int(y+h/2))
                if len(old_center_list)>0:
                    old_center=old_center_list[-1]
                    pixel_distance=np.sqrt((center[0]-old_center[0])**2+(center[1]-old_center[1])**2)
                    meters_distance=pixel_distance*pixel_to_meters_ratio
                    seconds_elapsed=1/fps_limit
                    speed=(meters_distance/seconds_elapsed)*3.6
                    vehicle_speeds.append(speed)
                    if len(vehicle_speeds)>1:
                        avg_speed=(vehicle_speeds[-1]+vehicle_speeds[-2])/2
                        cv2.putText(frame, f"{avg_speed:.2f} {speed_unit}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                old_center_list.append(center)

        if len(old_center_list)>fps_limit:
            old_center_list=old_center_list[-fps_limit:]

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()