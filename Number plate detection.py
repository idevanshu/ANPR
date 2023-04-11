import cv2
import os
import easyocr
import csv
 
visited = []
# opening the CSV file
with open('plates.csv', mode ='r') as file:
   
  # reading the CSV file
  csvFile = csv.reader(file)
 
  # displaying the contents of the CSV file
  for lines in csvFile:
        visited.append(lines[0])

frameWidth = 640
franeHeight = 480
plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
minArea = 500
reader = easyocr.Reader(['en'])
cap = cv2.VideoCapture("FILE LOCATION")
cap.set(3, frameWidth)
cap.set(4, franeHeight)
cap.set(10, 150)

# Make video copies
# mp4_files = [filename for filename in os.listdir(os.getcwd()) if filename.endswith('.mp4')]
# count_mp4_files = len(mp4_files)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(f'Video-{count_mp4_files}.mp4', fourcc, 20.0, (frameWidth, franeHeight))

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

    # out.write(img)

    # theft car recognition trigger
    if plateNumber.upper() in visited:
        print(f"Car detected with the number plate = {plateNumber.upper()}")
        break
        # extra functionanality needed

    if cv2.waitKey(1) & 0xFF == ord('s'):
        if result:
            cv2.imwrite(os.getcwd() + "\\" + str(count) + ".jpg", imgRoi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()