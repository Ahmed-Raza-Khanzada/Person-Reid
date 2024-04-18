from make_csv import get_random
import pandas as pd
from model import Model
import cv2
import torch
import numpy as np
from tqdm import tqdm
def make_embedding(model,img,DEVICE = "cuda"):

    # img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
    img = torch.from_numpy(cv2.resize(img, (48,128)).astype(np.float32)).permute(2, 0, 1) / 255.0
 

    model.eval()
    with torch.no_grad():
        img = img.to(DEVICE)
        img_enc = model(img.unsqueeze(0))
        img_enc = img_enc.detach().cpu().numpy().tolist()
    return img_enc
paths = ["/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/dukemtmc/bounding_box_train/","/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/market/Market-1501-v15.09.15/bounding_box_train/","/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/PKU-Reid-Dataset/PKUv1a_128x48/"]
df = pd.DataFrame(columns= ["img1_name","img2_name","img1","img2","label","dataset","path"])
total_images = 5000
model2 = Model()
model2.to("cuda" if torch.cuda.is_available() else "cpu")
model2.load_state_dict(torch.load("./model/reidmodel_final.pt"))
print("Embedding Model Loaded Succesfully")

for path in tqdm(paths):

    for images in tqdm(range(total_images)):

        a,p,n  = get_random(path)
   
        img1 = make_embedding(model2,cv2.imread(path+a))
        img2 = make_embedding(model2,cv2.imread(path+p))
        img3 = make_embedding(model2,cv2.imread(path+n))
    
        for j in range(2):
            if j==1:
                df = df._append({"img1_name":a,"img2_name":p,"img1":img1,"img2":img2,"label":j,"dataset":path.split("/")[-3].split("-")[0],"path":path},ignore_index=True)
            else:
                df = df._append({"img1_name":a,"img2_name":n,"img1":img1,"img2":img3,"label":j,"dataset":path.split("/")[-3].split("-")[0],"path":path},ignore_index=True)
print(len(df))
df = df.drop_duplicates(subset= ["img1_name","img2_name"])
max1 = 100
# for i in df.img1:
#     my_max = max(abs(max(i[0])) ,abs(min(i[0]))  )
#     if my_max>max1:
#         max1 = my_max
# for j in df.img2:
#     my_max = max(abs(max(j[0])) ,abs(min(j[0]))  )
#     if my_max>max1:
#         max1 = my_max
# print(max1,"Maimum value")
print(len(df))
df.img1 = df.img1.apply(lambda x: (np.array(x[0])/max1).tolist())
df.img2 = df.img2.apply(lambda x: (np.array(x[0])/max1).tolist())

print(df.head(10))
print(df.tail(10))

df.to_csv("../data/train_embeddings.csv",index = False)