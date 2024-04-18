import cv2
def get_face_coor(frame,mp_face_detection):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(frame_rgb)

    rects = []
    if results.detections:
        
        # print("Enter results detection")
        for detection in results.detections:
            x, y, w, h = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1]), \
                            int(detection.location_data.relative_bounding_box.ymin * frame.shape[0]), \
                            int(detection.location_data.relative_bounding_box.width * frame.shape[1]), \
                            int(detection.location_data.relative_bounding_box.height * frame.shape[0])

            x1, y1 = x + w, y + h
            rects.append(([x,y,w,h],detection.score[0],1)) #add "face"/1 also
        return rects
    
    return None