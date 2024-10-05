import cv2
import matplotlib.pyplot as plt
import os

harcascade = "model/haarcascade_russian_plate_number.xml"
if not os.path.exists('plates'):
    os.makedirs('plates')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

min_area = 500
count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_roi = img[y: y + h, x: x + w]

            plt.imshow(cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB))
            plt.title('Detected Plate')
            plt.show()

            cv2.imwrite(f"plates/scaned_img_{count}.jpg", img_roi)
            count += 1

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Result')
    plt.show()

cap.release()
