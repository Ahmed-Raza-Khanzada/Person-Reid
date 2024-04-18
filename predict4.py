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
    face_image = cv2.resize(face_image, (300, 300), fx=0.5, fy=0.5)
    
    face_encodings = face_recognition.face_encodings(face_image,known_face_locations=[(0, 300,300, 0)],num_jitters = 5,model = "large")

    return face_encodings[0]
def recognize_faces(known_face_encodings,face_encodings):

  


    if len(face_encodings) > 0:
        name = False
        # known_face_encodings,face_encodings = np.array(known_face_encodings),np.array(face_encodings).squeeze()
        # print(known_face_encodings.shape,face_encodings.shape)
        # matches = list(face_recognition.compare_faces(known_face_encodings, face_encodings, tolerance=0.99))
        # print(matches.count(True))
        # print("Total matches length",len(matches))
        # if matches.count(True)>6:#len(matches)//2:
        #     name = True
        c1=0
        for arr in known_face_encodings:
            match = euclidean_dist(face_encodings,arr)
            print("match",match)
            if match<400:
                c1+=1
            if c1>=len(known_face_encodings)//2:
                name = True
             
        return name,c1
   
    return False,c1



thresh_len = 50 # more better40
framethreshold = 5  #least better
framethreshold2 = 2 #least better
match_thresh =  3   #2
match_thresh2 = 4   #4
# l = [[1,9],[6,8,11],[4,7],[2],[3]]


all_features = {}
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
matched_ids = {}
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
                    track_id = result.id[0].astype(int)
                    track_id = str(track_id)
                    embedding = make_embedding(model=model2,img = frame[y1:y2,x1:x2]).squeeze()
                    cls = round(result.cls.item())
                    if len(all_features.keys())==0:
                        all_features[track_id] = [[embedding],((x1+x2)//2,(y1+y2)//2)]
                    else:

                        if track_id not in all_features.keys() :
                                old_matched = next((old_ids for old_ids,matched_lst in matched_ids.items() if track_id in matched_lst),None)
                                
                                if old_matched is None:
                                    finally_recognized = False
                                    temp = {}
                                    for old_id,features_value in all_features.items():
                                        old_embedding_list = features_value[0]
                                        print(track_id,old_id)
                                        recognized,c1 = recognize_faces(old_embedding_list,embedding)
                                        temp[old_id] = c1
                                    temp = dict(sorted(temp.items(),reverse=True,key = lambda item:item[1]))
                                    
                                    print(temp,"PPPPPPPPPPPPPPPPPPPPP")
                                    great =  list(temp.keys())[0]
                                    if temp[great]>0:
                                        old_id = great
                                        if old_id in all_ids:
                                            continue
                                        if old_id in matched_ids.keys():
                                            old_matched_ids = matched_ids[old_id]
                                            old_matched_ids.append(track_id)
                                            matched_ids[old_id] = old_matched_ids
                                        else:
                                            matched_ids[old_id] = [track_id]
                                        old_embedding_list.append(embedding)
                                        all_features[old_id] = [old_embedding_list,((x1+x2)//2,(y1+y2)//2)]
                                        
                                        track_id = old_id
                                        finally_recognized = True
                                        break
                                    if not finally_recognized:
                                        all_features[track_id] = [[embedding],((x1+x2)//2,(y1+y2)//2)]
                                else:
                                    track_id = old_matched
                        else:
                            count_embedd = len(all_features[track_id][0])
                            if count_embedd<10:
                                old_embedding_list = all_features[track_id][0]
                                old_embedding_list.append(embedding)
                                all_features[track_id][0] = old_embedding_list
                                
            
                    color = (0,0,255)
                    # color = (np.array(cmap(int(track_id if int(track_id)<10 else int(track_id)%10)))*255)[-2::-1].astype(np.uint8).tolist()
        
                    frame = draw_bbox(frame,x1,y1,x2,y2,color)
                  
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    color2= (0,0,0)
                    cv2.putText(frame, str(track_id), (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 2, cv2.LINE_AA)
            for result_f in result_face[0].boxes.cpu().numpy():
                x1_f, y1_f, x2_f, y2_f = result_f.xyxy.astype(int).tolist()[0]
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



