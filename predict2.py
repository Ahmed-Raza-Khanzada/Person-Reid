from ultralytics import YOLO
import cv2 
import numpy as np
import os
import time
from tqdm import tqdm
import pandas as pd
import torch
from Train_scripts.model import Model
from  match import make_embedding,make_matching

from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
model2 = Model()
model2.to("cuda" if torch.cuda.is_available() else "cpu")
model2.load_state_dict(torch.load("./Train_scripts/model/reidmodel_final.pt"))
print("Embedding Model Loaded Succesfully")

old_features =  {}

thresh_len = 50 # more better40
framethreshold = 5  #least better
framethreshold2 = 2 #least better
match_thresh =  3   #2
match_thresh2 = 4   #4
# l = [[1,9],[6,8,11],[4,7],[2],[3]]
def draw_bbox(image, x, y, x2,y2, color):
    frame = image.copy()
    box_coords = [(x, y), (x2,y), (x2, y2), (x, y2)]
    overlay = image.copy()
    transparent_box_alpha = 0.4
    cv2.fillPoly(overlay, [np.array(box_coords, np.int32)], color)
    cv2.addWeighted(overlay, transparent_box_alpha, image, 1, 0, image)
    frame[y:y2,x:x2] = image[y:y2,x:x2]
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    # image[y:y2,x:x2] = overlay
    return frame
def infer(path,f, model="Nano", u=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']):

    ids = set()

    model = YOLO("yolov8x.pt")
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    kwargs = {'device': 0, 'verbose': False,"conf":0.1,"classes":0,"tracker" :"botsort.yaml"}

    n = 0
    for results in model.track(source=path,persist=True,stream=True,**kwargs):
            frame = results.orig_img
            # print("Reults: ",results.boxes.cpu().numpy())
            # print(results)
            if results.boxes.id is not None:
                all_ids =results.boxes.id.cpu().numpy().astype(int)
            rectangle1 = {}
            for result in results.boxes.cpu().numpy():
                if result.id:
                    x1, y1, x2, y2 = result.xyxy.astype(int).tolist()[0]
                  
                    track_id = result.id[0].astype(int)
                    track_id = str(track_id)
                    if track_id in old_features.keys() :
                        old_features_this = old_features[track_id][1]
                        last_frame_this = old_features[track_id][0]
                        if len(old_features_this)<thresh_len  and n-last_frame_this>framethreshold2:
                            old_features_this.append(make_embedding(model=model2,img = frame[y1:y2,x1:x2]))
                            old_features[track_id] = (n,old_features_this)
                        # elif len(old_features_this)<thresh_len  and n-last_frame_this>framethreshold2:
                        #     old_features_this.append(make_embedding(model=model2,img = frame[y1:y2,x1:x2]))
                        #     old_features[track_id] = (n,old_features_this)
                        elif len(old_features_this)>=thresh_len  and n-last_frame_this>framethreshold:
                            old_features_this.pop(0)
                            old_features_this.append(make_embedding(model=model2,img = frame[y1:y2,x1:x2]))
                            old_features[track_id] = (n,old_features_this)
                    else:

                        new_embedding = make_embedding(model=model2,img = frame[y1:y2,x1:x2])
                        count_score = 0
                        check = {}
                        updated = False
                        for old_ids in old_features.keys():
                            old_features_this = old_features[old_ids][1]
                            # old_frame_this = old_features[old_ids][0]

                            for feature in old_features_this:
                                score = make_matching(feature,new_embedding)
                                print(score,old_ids,track_id)
                                if track_id=="8" and old_ids in ["6","11"]:
                                    print("score: ",score)

                                if score < match_thresh:
                                    if len(old_features_this)>=thresh_len:
                                        old_features_this.pop(0)
            
                                    old_features_this.append(new_embedding)
                                    old_features[old_ids] = (n,old_features_this)
                                    if old_ids not in all_ids:
                                        print(track_id,"current")
                                        track_id = old_ids
                                        print(track_id,"updated")
                                    updated = True
                                    break
                                elif score<match_thresh2:
                                    count_score +=1 
                                    if old_ids not in check.keys():
                                        check[old_ids] = [score,count_score]
                                    else:
                                        last_score = check[old_ids][0]
                                       
                                        check[old_ids] = [last_score+score,count_score]
                                if updated:
                                    break
                            if updated:
                                break
                        if not updated :
                            if len(check)>0:
                                check = dict(sorted(check.items(), key= lambda item: item[1][1],reverse=True))
                                if list(check.keys())[0] not in all_ids:
                                    track_id = list(check.keys())[0]
                                old_features_this = old_features[track_id][1]
                                if len(old_features_this)>=thresh_len:
                                    old_features_this.pop(0)
        
                                old_features_this.append(new_embedding)
                                old_features[track_id] = [n,old_features_this]
                                # updated = True
                            else:
                                old_features[track_id] = [n,[new_embedding]]
                    cls = round(result.cls.item())
                    rectangle1[track_id] = [cls,(x1,y1,x2,y2)]
            for track_id,(cls,bbox) in rectangle1.items():
                color = (0,0,255)
                # color = (np.array(cmap(int(track_id if int(track_id)<10 else int(track_id)%10)))*255)[-2::-1].astype(np.uint8).tolist()
       
                frame = draw_bbox(frame,x1,y1,x2,y2,color)
                x1,y1,x2,y2 = bbox
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                color2= (0,0,0)
                cv2.putText(frame, str(track_id), (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color2, 2, cv2.LINE_AA)
           

            n+=1
            yield frame



vid ="../Appearance Tracking/videoplayback.mp4" #"1099731453-preview.mp4" #
vid = "1099731453-preview.mp4"
vid = "./fr3/Fast 5_ Salute Mi Familia.mp4"
vid = "./fr3/shutterstock6.webm"
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



