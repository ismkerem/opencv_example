import  cv2
#video = cv2.VideoCapture("video/human_face.mp4")
video = cv2.VideoCapture(0) #webcam
face_cascade = cv2.CascadeClassifier("cascades/frontalface.xml")

while True:
    empty,frame=video.read()
    frame = cv2.flip(frame,1) #webcam
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #faces = face_cascade.detectMultiScale(gray,1.3,4)
    faces = face_cascade.detectMultiScale(gray, 1.4, 1) #webcam
    for (x,y,z,t) in faces:
        cv2.rectangle(frame,(x,y),(x+z,y+t),(0,255,0),2)
    cv2.imshow("image",frame)

    if cv2.waitKey(5) & 0xFF == ord('a'):
        break

video.release()
cv2.destroyAllWindows()