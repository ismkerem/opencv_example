import  cv2

video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("cascades/frontalface.xml")
eye_cascade = cv2.CascadeClassifier("cascades/eye.xml")

while 1:

    ret, frame = video.read()
    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        gray_roi = gray[y:y + h, x:x + w]
        frame_roi = frame[y:y + h, x:x + w]


        eyes = eye_cascade.detectMultiScale(gray_roi)


        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame_roi, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)


    cv2.imshow('video', frame)


    if cv2.waitKey(5) & 0xFF == ord('a'):
        break
video.release()
cv2.destroyAllWindows()