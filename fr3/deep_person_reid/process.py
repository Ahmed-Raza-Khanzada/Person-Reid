import face_recognition
import os
import cv2
import numpy as np
def process_face(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = cv2.resize(face_image, (300, 300), fx=0.5, fy=0.5)
    
    face_encodings = face_recognition.face_encodings(face_image,known_face_locations=[(0, 300,300, 0)],num_jitters = 5,model = "large")

    return face_encodings[0]
def recognize_faces(image,known_faces_encodings):

    face_encodings = process_face(image)

    # print(image.shape,face_encodings)

    # known_face_encodings = np.array(known_faces_encodings)

    if len(face_encodings) > 0:
        name = "Unknown"
        for personid, known_face_encodings in known_faces_encodings.items():
            # print(known_face_encodings.shape)
   

            
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings, tolerance=0.4)
            if matches.count(True)>len(matches)//2:
                name = personid
                break
            # if True in matches:
            #     # print(matches)
            #     # first_match_index = matches.index(True)
            #     name = personid
            #     print("Matched &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            #     break
        return name
   
    return False
