import cv2
import os
import tensorflow as tf
import numpy as np

frameWidth = 680
franeHeight = 480
fps = 20.0

plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
minArea = 500

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model('path_to_model.h5')

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3, frameWidth)
cap.set(4, franeHeight)
cap.set(10, 150)

mp4_files = [filename for filename in os.listdir(os.getcwd()) if filename.endswith('.mp4')]
file_name_count = len(mp4_files)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'Video-{file_name_count}.mp4', fourcc, fps, (frameWidth, franeHeight))

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
            imgRoi = cv2.resize(imgRoi, (224, 224))  # Resize the image to match the input shape of the model
            imgRoi = np.expand_dims(imgRoi, axis=0)  # Add a batch dimension
            imgRoi = imgRoi / 255.0  # Normalize the image
            result = model.predict(imgRoi)
            plateNumber = result[0]  # The model outputs the license plate number directly

            # Draw the license plate number on the image
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
