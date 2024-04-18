from ultralytics import YOLO
import cv2 
import numpy as np

import os

import time
from tqdm import tqdm
import pandas as pd



final = []
l = [[1,9],[6,8,11],[4,7],[2],[3]]


def infer(path,f, model="Nano", u=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']):

    ids = set()

    model = YOLO("yolov8x.pt")
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    kwargs = {'device': 0, 'verbose': False,"conf":0.20,"classes":0}

   
    n = 0
    for results in model.track(source=path,persist=True,stream=True, **kwargs):
            frame = results.orig_img
            # print("Reults: ",results.boxes.cpu().numpy())
            # print(results)


            for result in results.boxes.cpu().numpy():
                if result.id:
                    x1, y1, x2, y2 = result.xyxy.astype(int).tolist()[0]
                    track_id = result.id[0].astype(int)
                    # track_id = str(track_id)
                    for pos,l1 in enumerate(l):
                         if track_id in l1:
                                if not os.path.exists("./data/"+str(pos)):
                                    os.makedirs("./data/"+str(pos),exist_ok=True)
                                final.append("./data/"+str(pos)+"/"+str(track_id)+"_"+str(n)+".jpg")
                                cv2.imwrite("./data/"+str(pos)+"/"+str(track_id)+"_"+str(n)+".jpg", frame[y1:y2,x1:x2])
            n+=1
            yield frame
t_time = time.time()
vid = "1099731453-preview.mp4"

start_time = time.time()
if vid.endswith(".mp4"):
    cap = cv2.VideoCapture(vid)
    h,w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    w,h = int(w),int(h)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count1  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    # wrt2 = cv2.VideoWriter("./predict_custom_all_extra_final.mp4", cv2.VideoWriter.fourcc(*"mp4v"), 60, (w, h))
    for frame in tqdm(infer(f"./{vid}",fps),total=count1):
         pass
    df = pd.DataFrame()
    df["images"] = np.array(final)

    df.to_csv("./data/my_data.csv",index=False)