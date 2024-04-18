import numpy as np
import cv2
import mediapipe as mp
import time
from get_face import get_face_coor
from process import process_face
import os
import uuid  # Import for generating random IDs

mp_face_detection = mp.solutions.face_detection.FaceDetection()

cap = cv2.VideoCapture(0)
os.makedirs('encodings', exist_ok=True)
frame_thresh = 10
encodings = []  
person_id = str(uuid.uuid4()) 
n = 0
last_thresh=30
last = None
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        rects = get_face_coor(frame, mp_face_detection)
        # print(rects)
        if rects:
            for rect in rects:
                print(rect[0])
                x, y, w, h = rect[0]
                x1, y1 = x + w, y + h
                if (last is None or n-last >= last_thresh  ):
                    face_encoding = process_face(frame[y:y1, x:x1])
                    if len(face_encoding)>0:
                        encodings.append(face_encoding)
                        cv2.putText(frame, f"Captured", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        last = n
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, person_id, (x,max(0,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
      
        
        if len(encodings) >= frame_thresh: 
            break

    except Exception as e:
        print(e)
        continue
    n+=1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


encodings_array = np.array(encodings)
np.save(os.path.join('encodings', f'{person_id}.npy'), encodings_array)
print(f"All encodings saved for person {person_id} (shape: {encodings_array.shape})")

cap.release()
cv2.destroyAllWindows()
