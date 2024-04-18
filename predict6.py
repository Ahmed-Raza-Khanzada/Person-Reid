from ultralytics import YOLO
import cv2 
import numpy as np
import os
import time
from tqdm import tqdm
import pandas as pd
import torch
import math
from Train_scripts.model import Model
from  match import make_embedding,make_matching
import face_recognition
from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
model2 = Model()
model2.to("cuda" if torch.cuda.is_available() else "cpu")
model2.load_state_dict(torch.load("./Train_scripts/model/reidmodel_final.pt"))
print("Embedding Model Loaded Succesfully")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_track_embeddings(track_embeddings, n_components=2, title="Track Embeddings"):
  """
  Plots all track IDs' embeddings using PCA for visualization.

  Args:
      track_embeddings: A dictionary where keys are track IDs and values are lists of embeddings.
      n_components: Number of principal components to use for visualization (default: 2).
      title: Title for the plot (default: "Track Embeddings").
  """

  # Collect all embeddings into a single list
  all_embeddings = []
  for track_id, embeddings in track_embeddings.items():
    all_embeddings.extend(embeddings)

  # Apply PCA for dimensionality reduction
  pca = PCA(n_components=n_components)
  reduced_embeddings = pca.fit_transform(all_embeddings)

  # Create scatter plot with different colors for each track ID
  colors = plt.cm.tab10(range(len(track_embeddings)))  # Choose 10 colors from the tab10 colormap
  markers = ['o', 's', '^', 'P', 'x', 'D', 'v', '<', '>', 'p']  # Choose 10 markers

  for i, (track_id, embeddings) in enumerate(track_embeddings.items()):
    reduced_embedding_subset = reduced_embeddings[sum(len(l) for l in track_embeddings.values())[:i]:sum(len(l) for l in track_embeddings.values())[:i+1]]
    plt.scatter(reduced_embedding_subset[:, 0], reduced_embedding_subset[:, 1], label=track_id, c=colors[i % len(colors)], marker=markers[i % len(markers)])

  # Add labels and title
  plt.xlabel("Principal Component 1")
  plt.ylabel("Principal Component 2")
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()

def euclidean_dist(img_enc, anc_enc_arr):
    dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T))
    return dist
def process_face(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = cv2.resize(face_image, (48, 48), fx=0.5, fy=0.5)
    
    face_encodings = face_recognition.face_encodings(face_image,known_face_locations=[(0, 48,48, 0)],num_jitters = 2,model = "large")

    return face_encodings[0]

def recognize_faces(known_face_encodings,face_encodings):

        # known_face_encodings = np.array(known_face_encodings)
        # print(known_face_encodings.shape,face_encodings.shape)
        known_face_encodings = [x for x in known_face_encodings if x is not None]
        print(face_encodings.shape)
      
        if len(known_face_encodings)==0:
            return 0
        matches = list(face_recognition.compare_faces(known_face_encodings, face_encodings, tolerance=0.5))
        return matches.count(True)


def find_last(arr,col):
    nzc = np.count_nonzero(arr!=col,axis=1)
    print(nzc)
    if nzc[0] ==0:
        last_idx =-1
    else:
        last_idx = arr.shape[0]-nzc[0]
    return last_idx

def make_decesion(s1,s2,thresh = 5):
    if (s1+s2)//2>thresh:
        return True
    return False
def recognize_person(known_persons_encodings,person_encoding):

  


    if len(person_encoding) > 0:
        name = False

        c1=0
        for arr in known_persons_encodings:
            match = euclidean_dist(person_encoding,arr)
            print("match",match)
            if match<400:
                c1+=1
            if c1>=len(known_persons_encodings)//2:
                name = True
             
        return name,c1
   
    return False,c1



thresh_len = 50 # more better40
framethreshold = 5  #least better
framethreshold2 = 2 #least better
match_thresh =  3   #2
match_thresh2 = 4   #4
total_length = 10
# l = [[1,9],[6,8,11],[4,7],[2],[3]]


all_features = {}
matched_ids = {}
def draw_bbox(image, x, y, x2,y2, color,transparent_box_alpha = 0.4):
    frame = image.copy()
    box_coords = [(x, y), (x2,y), (x2, y2), (x, y2)]
    overlay = image.copy()
    
    cv2.fillPoly(overlay, [np.array(box_coords, np.int32)], color)
    cv2.addWeighted(overlay, transparent_box_alpha, image, 1, 0, image)
    frame[y:y2,x:x2] = image[y:y2,x:x2]
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    # image[y:y2,x:x2] = overlay
    return frame
def cal_dist(cent1,cent2):
    return math.sqrt((cent1[0]-cent2[0])**2+(cent1[1]-cent2[1])**2)

def infer(path,f, model="Nano", u=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']):


    facemodel = YOLO('yolov8n-face.pt')

    model = YOLO("yolov8x.pt")
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    kwargs = {'device': 0, 'verbose': False,"conf":0.1,"classes":0,"tracker" :"botsort.yaml"}

    n = 0
    for results in model.track(source=path,persist=True,stream=True,**kwargs):
            origframe = results.orig_img
            frame = origframe.copy()
            # print("Reults: ",results.boxes.cpu().numpy())
            # print(results)
            result_face = facemodel.track(source=frame,persist=True,**kwargs)
            
                

            if results.boxes.id is not None:
                all_ids =results.boxes.id.cpu().numpy().astype(int)
            for result in results.boxes.cpu().numpy():
                if result.id:
                    x1, y1, x2, y2 = result.xyxy.astype(int).tolist()[0]
                    face_embbeding =None
                    for result_f in result_face[0].boxes.cpu().numpy():
                        x1_f, y1_f, x2_f, y2_f = result_f.xyxy.astype(int).tolist()[0]
                        center_facex,center_facey = ((x1_f+x2_f)//2),  ((y1_f+y2_f)//2)
                        if (center_facex>x1 and center_facex<x2) and (center_facey>y1 and center_facey<y2): 
                            face_embbeding = process_face(frame[y1_f:y2_f,x1_f:x2_f])
                            break
                    
                    track_id = result.id[0].astype(int)
                    track_id = str(track_id)
                    person_embedding = make_embedding(model=model2,img = frame[y1:y2,x1:x2]).squeeze()
                    
                
                    cls = round(result.cls.item())
                    if len(all_features.keys())==0:
                        # arr = np.zeros((total_length,2))
                        # print(arr,person_embedding.shape,face_embbeding.shape)
                        # arr[0][0],arr[0][1] = person_embedding,face_embbeding
                
                        all_features[track_id] = [[person_embedding],[face_embbeding]]#arr
                    else:
                        if track_id not in all_features.keys():
                            old_matched = next((old_ids for old_ids,matched_lst in matched_ids.items() if track_id in matched_lst),None)                    
                            if old_matched is None:
                                 
                                    temp = {}
                                    temp2={}
                                    for old_id,features_value in all_features.items():
                                        
                                        old_personembedding_list = features_value[0]#[x[0] for x in features_value]#features_value[:,0]
                                        old_faceembedding_list = features_value[1]#[x[1] for x in features_value]#features_value[:,1]
                                        print(track_id,old_id)
                                        # print(old_personembedding_list[0].shape,old_faceembedding_list[0].shape)
                                        recognized,c1 = recognize_person(old_personembedding_list,person_embedding)
                                        if len(old_faceembedding_list)>0  and face_embbeding is not None:
                                            c2 = recognize_faces(old_faceembedding_list,face_embbeding)
                                        else:
                                            c2 = 0
                                        temp[track_id] = c1
                                        temp2[track_id] = c2
                                        temp = dict(sorted(temp.items(),reverse=True,key = lambda item:item[1]))
                                        temp2 = dict(sorted(temp2.items(),reverse=True,key = lambda item:item[1]))
                                    s1_personkey,s2_facekey = list(temp.keys())[0],list(temp2.keys())[0]
                                    if s1_personkey==s2_facekey:
                                        res = make_decesion(temp[s1_personkey],temp2[s2_facekey],thresh = 5)
                                        print(res)
                                        if res:
                                            track_id = s1_personkey
                                            if matched_ids.get(s1_personkey) is not None:
                                                old_l = matched_ids[s1_personkey]
                                                old_l.append(track_id)
                                                matched_ids[s1_personkey] = old_l 
                                            else:
                                                matched_ids[s1_personkey] = [track_id]
                                    else:
                                        face_score = temp[s2_facekey] *0.6
                                        person_score = temp2[s1_personkey] *0.4
                                        if person_score>face_score:
                                            track_id = s1_personkey
                                            if matched_ids.get(s1_personkey) is not None:
                                                old_l = matched_ids[s1_personkey]
                                                old_l.append(track_id)
                                                matched_ids[s1_personkey] = old_l 
                                            else:
                                                matched_ids[s1_personkey] = [track_id]
                                        else:
                                            track_id = s2_facekey
                                            if matched_ids.get(s2_facekey) is not None:
                                                old_l = matched_ids[s2_facekey]
                                                old_l.append(track_id)
                                                matched_ids[s2_facekey] = old_l 
                                            else:
                                                matched_ids[s2_facekey] = [track_id]
                            else:
                                track_id = old_matched

                        else:
                            features_arr = all_features[track_id] 
                            
                            # last_idx_person,last_idx_face = find_last(features_arr,0),find_last(features_arr,1)      
                            # if person_embedding:      
                            #     if last_idx_person ==total_length-1:
                            #         features_arr[0][0] = person_embedding
                            #     else:
                            #         features_arr[last_idx_person][0] = person_embedding
                            # if face_embbeding:
                            #     if last_idx_face==total_length-1:
                            #         features_arr[0][1] = face_embbeding
                            #     else:
                            #         features_arr[last_idx_face][1] = face_embbeding
                            features_arr_person,features_arr_face = features_arr[0],features_arr[1]
                            if len(features_arr_person)==total_length:
                                features_arr_person.pop(0)
                                features_arr_person.append(person_embedding)
                            else:
                                features_arr_person.append(person_embedding)
                            if len(features_arr_face)==total_length:
                                features_arr_face.pop(0)
                                features_arr_face.append(face_embbeding)
                            else:
                                features_arr_face.append(face_embbeding)        
                            features_arr[0],features_arr[1] = features_arr_person,features_arr_face
                            all_features[track_id] = features_arr
                    color = (0,0,255)
                    # color = (np.array(cmap(int(track_id if int(track_id)<10 else int(track_id)%10)))*255)[-2::-1].astype(np.uint8).tolist()
        
                    frame = draw_bbox(frame,x1,y1,x2,y2,color)
                
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    color2= (0,0,0)
                    cv2.putText(frame, str(track_id), (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 2, cv2.LINE_AA)
                    frame = draw_bbox(frame,x1_f,y1_f,x2_f,y2_f,(0,255,255),0.1)
            n+=1
            yield frame



vid = "./fr3/Fast 5_ Salute Mi Familia.mp4"
vid ="../Appearance Tracking/videoplayback.mp4" #"1099731453-preview.mp4" #
vid = "./fr3/shutterstock6.webm"
vid = "1099731453-preview.mp4"
start_time = time.time()

if vid.endswith(".mp4") or vid.endswith(".webm"):
    cap = cv2.VideoCapture(vid)
    h,w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    w,h = int(w),int(h)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    wrt2 = cv2.VideoWriter("./output/predict_custom_all_check_all_office2.mp4", cv2.VideoWriter.fourcc(*"mp4v"), 40, (w, h))
    for frame in tqdm(infer(f"./{vid}",fps),total=count_frame):
        cv2.imshow("Frame",frame)
        cv2.waitKey(1)
        # wrt2.write(frame)
    print(time.time()-start_time)
    print(vid ,"Done")



