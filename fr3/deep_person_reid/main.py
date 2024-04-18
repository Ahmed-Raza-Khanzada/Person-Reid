import cv2 
import numpy as np
from process import process_face
from get_face import get_face_coor
import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=200,embedder="torchreid")


mp_face_detection = mp.solutions.face_detection.FaceDetection()
v = "/home/ahmedrazakhanzada/Downloads/1099731479-preview.mp4"
cap = cv2.VideoCapture(v)
opt =   {"device":"cpu","track_high_thresh":0.6,"track_low_thresh":0.1,"new_track_thresh":0.7,"track_buffer":30,"with_reid":False,"proximity_thresh":0.5,"appearance_thresh":0.25,"ablation":False,"name":"MOT20-08","cmc_method":"file"}



while True:
    ret,frame = cap.read()
    if not ret:
        break #continue
    try:
        rects = get_face_coor(frame, mp_face_detection)
        
        if rects:
            # rects = np.array(rects,dtype = np.float16)
            # rects = np.insert(rects, -1, np.array([1]*len(rects)), axis=1)
          
            tracks = tracker.update_tracks(rects, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
          
            for track in tracks:
                
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
            
                x, y, x1, y1= ltrb
                if x<0:
                    continue
                if y<0:
                    continue
                if x1>frame.shape[1]:
                    continue
                if y1>frame.shape[0]:
                    continue
                x,y,x1,y1 = int(x),int(y),int(x1),int(y1)
                color = (0,0,255)
                cv2.line(frame, (x, y), (x +10, y), tuple(color), 6)  # TL
                cv2.line(frame, (x, y), (x, y +10), tuple(color), 6)

                cv2.line(frame, (x1, y), (x1 -10, y), tuple(color), 6)  # TR
                cv2.line(frame, (x1, y), (x1, y +10), tuple(color), 6)

                cv2.line(frame, (x, y1), (x +10, y1), tuple(color), 6)  # BL
                cv2.line(frame, (x, y1), (x, y1 -10), tuple(color), 6)

                cv2.line(frame, (x1, y1), (x1 -10, y1), tuple(color), 6)  # BR
                cv2.line(frame, (x1, y1), (x1, y1 -10), tuple(color), 6)
                # cv2.rectangle(frame, (x, y), (x1, y1), tuple(color), 2)
                color2 = list(color)[::-1]
                
                cv2.putText(frame, str(track_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,color2 , 2)
                face_encoding = process_face(frame[y:y1, x:x1])
             
    except Exception as e:
        print(e)
        continue


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
