from ultralytics import YOLO
import cv2 
import numpy as np
import os
import time
from tqdm import tqdm
from torch import nn
import pandas as pd
import torch
from Train_scripts.model import Model
from  match import make_embedding,make_matching

from matplotlib.cm import get_cmap

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=512):
        super(SiameseNetwork, self).__init__()

        self.shared_nn = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size)  
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
       
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        img1_embedding = img1#self.shared_nn(img1)
        img2_embedding =img2 #self.shared_nn(img2)

        concatenated = torch.cat((img1_embedding, img2_embedding), dim=1)
        
      
        output = self.fc(concatenated)
        return output

cmap = get_cmap('tab10')
model2 = Model()
model2.to("cuda" if torch.cuda.is_available() else "cpu")
model2.load_state_dict(torch.load("./Train_scripts/model/reidmodel_final.pt"))


classifier = SiameseNetwork(512)
classifier.to("cuda" if torch.cuda.is_available() else "cpu")
classifier.load_state_dict(torch.load("./Train_scripts/model/classifier.pt"))
print("Embedding Model Loaded Succesfully")

old_features =  {}

thresh_len = 40 # more better40
framethreshold = 2  #least better
framethreshold2 = 2 #least better
match_thresh = 2    #2
match_thresh2 = 4   #4

map_ids = {}
continue_tracked = {}
len_thresh = 10
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
def predict_emb(classifier, img1, img2, threshold=0.5):
    classifier.eval()
    with torch.no_grad():
        img1, img2 = img1.float(), img2.float()
        output = classifier(torch.cat((img1, img2), dim=1))
        prediction = (output > threshold).float()
    return prediction.item()
def infer(path,f, model="Nano", u=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']):

    ids = set()

    model = YOLO("yolov8x.pt")
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    kwargs = {'device': 0, 'verbose': False,"conf":0.1,"classes":0,"tracker" :"botsort.yaml"}

    recent_embeddings = {}
    n = 0
    for results in model.track(source=path,persist=True,stream=True,**kwargs):
            frame = results.orig_img
            # print("Reults: ",results.boxes.cpu().numpy())
            # print(results)
            if results.boxes.cpu().numpy():
                all_ids =results.boxes.id.cpu().numpy().astype(int)
            rectangle1 = {}
            for result in results.boxes.cpu().numpy():
                if result.id:
                    x1, y1, x2, y2 = result.xyxy.astype(int).tolist()[0]
                  
                    track_id = result.id[0].astype(int)
                    track_id = str(track_id)

                    cls = round(result.cls.item())
                    rectangle1[track_id] = [cls,(x1,y1,x2,y2)]
                    color = (0,0,255)
                    cx,cy = (int((x1+x2)/2),int((y1+y2)/2))
                    current_embbedding = make_embedding(model,frame[y1:y2,x1:x2])
                    if track_id not in recent_embeddings.keys():

                        match1 = False
                        current_embbedding = make_embedding(model,frame[y1:y2,x1:x2])
                        for old_emb in old_features.keys():
                            old_emb = old_features[old_emb]
                            ret = predict_emb(classifier,current_embbedding,old_emb,thresh_len)
                            if ret:
                                match1 = True
                                break
                        if match1:
                            if track_id in continue_tracked.keys():
                                continue_tracked[track_id] +=1
                            else:
                                continue_tracked[track_id] = 1
                                
                            map_ids[old_emb] = track_id
                        else:
                            recent_embeddings[track_id] = current_embbedding
                        
                    else:
                        if len(recent_embeddings[track_id])<len_thresh:
                            recent_embeddings[track_id].append(make_embedding(model,frame[y1:y2,x1:x2]))
                        else:
                            recent_embeddings[track_id] = recent_embeddings[track_id][1:]
                            recent_embeddings[track_id].append(make_embedding(model,frame[y1:y2,x1:x2]))

                    cv2.putText(frame, str(track_id), (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 2, cv2.LINE_AA)

            n+=1
            yield frame



vid = "1099731453-preview.mp4"
vid ="../Appearance Tracking/videoplayback.mp4" #"1099731453-preview.mp4" #
start_time = time.time()
if vid.endswith(".mp4"):
    cap = cv2.VideoCapture(vid)
    h,w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    w,h = int(w),int(h)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    wrt2 = cv2.VideoWriter("./output/predict_custom_all_check_all_office_test.mp4", cv2.VideoWriter.fourcc(*"mp4v"), 10, (w, h))
    for frame in tqdm(infer(f"./{vid}",fps),total=count_frame):
        # cv2.imshow("Frame",frame)
        # cv2.waitKey(1)
        wrt2.write(frame)
    print(time.time()-start_time)
    print(vid ,"Done")



