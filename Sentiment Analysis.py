
import cv2
import numpy
#pip install openCV on command prompt #pip install numpy
from deepface import DeepFace
import matplotlib.pyplot as plt
img = cv2.imread('sad woman.jpg')
#pip install deepface #import matplotlib library
#reads the given image
plt.imshow(img) #shows image in BGR
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) #converts img format from BGR to
RGB
predictions = {0} #declares datatype dictionary
predictions = DeepFace.analyze(img) #calls function analyze from DeepFace
predictions
type(predictions)
d=predictions[0]
#display predictions
#shows datatype of variable predictions
#changind datatype from list to dict.
print(d["dominant_emotion"]) #extracting prediction type from dictionary
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml') #extracts facial features #haar cascade is a face recognition algorithm
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) faces = faceCascade.detectMultiScale(gray,1.1,4) for(x,y,w,h) in faces:
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) converted imag
#convert image to gray image #detects face
# color and size of rectangle around face #show the rectangle around face on
font = cv2.FONT_HERSHEY_SIMPLEX #type of font
cv2.putText(img,d['dominant_emotion'],(50,50),font,2,(0,255,0),2,cv2.LINE_4); #putText() method to display text
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
48 12002040501003
102046705 Computer Vision and Image Processing
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #extracts facial features
cap = cv2.VideoCapture(0) #turns on the webcam by index 0 #cap is capture variable
if not cap.isOpened(): #acknowlege
cap = cv2.VideoCapture(1) #else open by 1
if not cap.isOpened():
raise IOError("Unable to open webcam")
while True:
ret,frame = cap.read() #reads single image (frame) from video
result = DeepFace.analyze(frame) #.analyze() method to extract details d=result[0]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to gray image faces = faceCascade.detectMultiScale(gray,1.1,4) #detect face
for(x,y,w,h) in faces: #form a rectangle over face
cv2.rectangle(frame,(x, y),(x+w,y+h),(0,255,0),2) #attributes of rectangle
font = cv2.FONT_HERSHEY_SIMPLEX #font to display dominant emtions cv2.putText(frame,d['dominant_emotion'],(50,50),font,3,(0,255,0),2,cv2.LINE_4);
cv2.imshow('demo video',frame)
if cv2.waitKey(2) & 0xFF == ord('q'):
break cap.release()
cv2.destroyAllWindow()