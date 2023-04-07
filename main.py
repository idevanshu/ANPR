import cv2
import os
import easyocr

frameWidth = 640
franeHeight = 480
plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
minArea = 500
reader = easyocr.Reader(['en'])
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, frameWidth)
cap.set(4, franeHeight)
cap.set(10, 150)

# Define the codec for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # change codec to mp4v
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frameWidth, franeHeight)) # change file extension to .mp4

count = 0

while True:
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

    # Write the frame to the output video file
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        if result:
            cv2.imwrite(os.getcwd() + "\\" + str(count) + ".jpg", imgRoi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file, output file, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
