import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm

def get_random(path):
    all_images = os.listdir(path)
    
    anchor  = random.choice(all_images)
    
    positives = list(filter(lambda x: True if x.split("_")[0]==anchor.split("_")[0] else False,all_images))

    negative = random.choice(list(set(all_images)-set(positives)))
    if len(positives) >1:
        positive = random.choice(list(set(positives)-set([anchor])))
    else:
        positive = random.choice(positives)

    # print("Anchor: ",anchor)
    # print("Positive: ",positive)
    # print("Negative: ",negative)
    return anchor,positive,negative

# df = pd.DataFrame(columns=["Anchor","Positive","Negative","Dataset","Path"])

# length =4000
# paths = ["/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/dukemtmc/bounding_box_train/","/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/market/Market-1501-v15.09.15/bounding_box_train/","/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/PKU-Reid-Dataset/PKUv1a_128x48/"]
# for X in paths:
#     for image in tqdm(range(length),total=length):
#         anchor,positive,negative = get_random(X)
#         df = df._append({"Anchor":anchor,"Positive":positive,"Negative":negative,"Dataset":X.split("/")[-3].split("-")[0],"Path":X},ignore_index=True)
# print(df.head(10))
# print(len(df))
# print(df.tail(10))