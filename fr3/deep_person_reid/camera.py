import cv2
from BoT_SORT.tools.process import process_image
import time

 # Encode the known faces

class Video(object):
    def __init__(self,known_face_encoding,known_face_names,mp_face_detection):
        self.video = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.known_face_encodings=known_face_encoding
        self.known_face_names = known_face_names
        self.mp_face_detection  = mp_face_detection
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        

        ret, frame = self.video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(frame_rgb)
        # print("Read Camera")
        faces_names = []
        rects = []
        if results.detections:
            
            # print("Enter results detection")
            for detection in results.detections:
                x, y, w, h = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1]), \
                             int(detection.location_data.relative_bounding_box.ymin * frame.shape[0]), \
                             int(detection.location_data.relative_bounding_box.width * frame.shape[1]), \
                             int(detection.location_data.relative_bounding_box.height * frame.shape[0])

                x1, y1 = x + w, y + h
                try:
                    frame,face_name = process_image((x, y, w, h), frame,self.known_face_encodings,self.known_face_names)
                except Exception as e:
                    face_name = False
                faces_names.append(face_name)
                rects.append((x,y,x1,y1))
                
        
        
        return frame,faces_names,rects
