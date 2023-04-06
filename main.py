import cv2
import imutils
import numpy as np
import os

CASCADE_FILE = 'haarcascade_russian_plate_number.xml'

cascade = cv2.CascadeClassifier(CASCADE_FILE)

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        roi = frame[y:y + h, x:x + w]

        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

            if len(approx) == 4:
                area = cv2.contourArea(cnt)

                if area < 1000 or area > 10000:
                    continue

                cv2.drawContours(roi, [cnt], 0, (0, 255, 0), 2)

                plate = close[y:y + h, x:x + w]

                plate = cv2.resize(plate, (0, 0), fx=2, fy=2)
                plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

                plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                plate_contours, plate_hierarchy = cv2.findContours(plate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for plate_cnt in plate_contours:
                    plate_approx = cv2.approxPolyDP(plate_cnt, 0.02 * cv2.arcLength(plate_cnt, True), True)

                    if len(plate_approx) == 4:
                        plate_area = cv2.contourArea(plate_cnt)

                        if plate_area < 100 or plate_area > 1000:
                            continue

                        plate_rect = cv2.minAreaRect(plate_cnt)
                        plate_box = cv2.boxPoints(plate_rect)
                        plate_box = np.int0(plate_box)

                        cv2.drawContours(plate, [plate_box], 0, (0, 255, 0), 2)

                        plate_text = pytesseract.image_to_string(plate_thresh, config='--psm 11')

                        if len(plate_text) > 0:
                            print(f'License plate: {plate_text.strip()}')

    return frame

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = process_frame(frame)

    cv2.imshow('License Plate Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
