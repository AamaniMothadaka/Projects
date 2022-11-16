import mediapipe as mp
import numpy as np
import cv2
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

cap=cv2.VideoCapture(0)
# To capture video from webcam of lappy
name=input("Enter the name of the data:")
holistic=mp.solutions.holistic
hands=mp.solutions.hands
holis=holistic.Holistic()
drawing=mp.solutions.drawing_utils
x=[]
data_size=0
while True:
    lst=[]
    _, frm =cap.read()# To read the frame
    frm=cv2.flip(frm,1)
    res=holis.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        x.append(lst)
        data_size=data_size+1
            
    drawing.draw_landmarks(frm,res.face_landmarks,holistic.FACEMESH_TESSELATION)
    drawing.draw_landmarks(frm,res.left_hand_landmarks,hands.HAND_CONNECTIONS)   
    drawing.draw_landmarks(frm,res.right_hand_landmarks,hands.HAND_CONNECTIONS)
    cv2.putText(frm,str(data_size),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("window",frm) #To show it to user
    if cv2.waitKey(1)==27 or data_size>99: # if user preess escape key destroy all windows
        cv2.destroyAllWindows()
        cap.release()  #release the resource
        break
np.save(f"{name}.npy",np.array(x))
print(np.array(x).shape)
