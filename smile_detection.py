import cv2

img = cv2.imread("images/smile_face.jpg")
face_cascade = cv2.CascadeClassifier("cascades/frontalface.xml")
smile_cascade = cv2.CascadeClassifier("cascades/smile.xml")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face = face_cascade.detectMultiScale(gray,1.3,5)

for (x,y,z,t) in face:
    cv2.rectangle(img,(x,y),(x+z,y+t),(255,0,0),3)

det_img = img[y:y+t,x:x+t]
det_gray = gray[y:y+t,x:x+t]

smile = smile_cascade.detectMultiScale(det_gray,1,3)

for (x,y,z,t) in smile:
    cv2.rectangle(img,(x,y),(x+z,y+t),(0,255,0),2)






cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()