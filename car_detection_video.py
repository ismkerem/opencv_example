import cv2

vid = cv2.VideoCapture("video/car.mp4")
car_cascade = cv2.CascadeClassifier('cascades/car.xml')

while True:
    ret, frame = vid.read()
    frame = cv2.resize(frame, (700, 700))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, z, t) in cars:
        cv2.rectangle(frame, (x, y), (x + z, y + t), (0, 0, 255), 3)

    cv2.imshow('image', frame)

    if cv2.waitKey(5) & 0xFF == ord('a'):
        break

vid.release()
cv2.destroyAllWindows()
