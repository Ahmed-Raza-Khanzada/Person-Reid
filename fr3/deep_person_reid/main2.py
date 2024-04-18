from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=3,embedder="torchreid")
from process import process_face, recognize_faces
import os
import uuid
model = YOLO('yolo_weights/yolov8n-face.pt')

kwargs = {'device':"cpu", 'verbose': False,"conf":0.1,"classes":0,"tracker" :"botsort.yaml"}
path ="/home/ahmedrazakhanzada/Downloads/face-detection-yolov8-main/demo.mp4" 
path  = 0#"/home/ahmedrazakhanzada/Downloads/1099731479-preview.mp4"
path = "../shutterstock6.webm"
path ="/home/ahmedrazakhanzada/Downloads/Fast 5_ Salute Mi Familia.mp4" 


n = 0
def make_convert(l):
    l = np.array(l)
    l[:2] = l[:2] - l[2:] // 2
    return l
d= {}

encodings_directory = 'encodings'
known_encodings_list = {}
for file_name in os.listdir(encodings_directory):
    if file_name.endswith('.npy'):
        person_id = file_name.split('.')[0]
        encodings_array = np.load(os.path.join(encodings_directory, file_name))
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOO",encodings_array.shape)
        known_encodings_list[person_id] = encodings_array

current_face_encodings = {}

assigned_ids = {}
new_track_thresh = 15
for results in model.track(source=path,persist=True,stream=True,**kwargs):
        frame = results.orig_img

        if results.boxes.id is not None:
            rects = results.boxes.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            rects = [(make_convert(rect.xywh.astype("int").tolist()[0]),float(rect.conf.astype("float")),int(rect.cls.astype("int")))  for rect in rects]


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
                track_id = str(track_id)
                print("Track ID",track_id)
                if track_id not in assigned_ids.keys():
                    if track_id not in d.keys():
                        
                        known_name =  recognize_faces(frame[y:y1,x:x1],known_encodings_list)
                        print(known_name,'||||||||||||||||||||||||||||||||||||||||||||||')
                        if known_name !="Unknown":
                            if known_name not in assigned_ids.values():
                                assigned_ids[track_id] = known_name
                            else:
                                assign_idx = list(assigned_ids.values()).index(known_name)
                                track_id = list(assigned_ids.keys())[assign_idx]
                        else:
                        
                            d[track_id] = [[process_face(frame[y:y1,x:x1])],0]
                    else:
                        known_name =  recognize_faces(frame[y:y1,x:x1],known_encodings_list)
                        if known_name !="Unknown":
                            if known_name not in assigned_ids.values():
                                assigned_ids[track_id] = known_name

                            else:
                                assign_idx = list(assigned_ids.values()).index(known_name)
                                track_id = list(assigned_ids.keys())[assign_idx]
                        else:
                            if d[track_id][1] >=new_track_thresh:
                                person_id1 = str(uuid.uuid4())
                             
                                known_encodings_list[person_id1] = np.array(d[track_id][0])
                                assigned_ids[track_id] = person_id1
                                d.pop(track_id)
                                
                            else:
                                print(track_id,"start Tracking")
                                d[track_id][0].append(process_face(frame[y:y1,x:x1]))
                                d[track_id][1]+=1
                # print(d)
                # print(assigned_ids)
                color= (0, 255, 255)
                # cls = round(result.cls.item())

                cv2.line(frame, (x, y), (x +10, y), tuple(color), 1)  # TL
                cv2.line(frame, (x, y), (x, y +10), tuple(color), 1)

                cv2.line(frame, (x1, y), (x1 -10, y), tuple(color), 1)  # TR
                cv2.line(frame, (x1, y), (x1, y +10), tuple(color), 1)

                cv2.line(frame, (x, y1), (x +10, y1), tuple(color), 1)  # BL
                cv2.line(frame, (x, y1), (x, y1 -10), tuple(color), 1)

                cv2.line(frame, (x1, y1), (x1 -10, y1), tuple(color), 1)  # BR
                cv2.line(frame, (x1, y1), (x1, y1 -10), tuple(color), 1)
                # cv2.rectangle(frame, (x, y), (x1, y1), tuple(color), 2)
                color2 = (0,0,255)#list(color)[::-1]
                
                cv2.putText(frame, str(track_id), (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9,color2 , 2)

        cv2.imshow("Frame", cv2.resize(frame,(900,500)))
        n+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break