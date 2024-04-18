import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from model import Model
from train import train_model
from eval import eval_model
from Dataloader import Load_Data
from torch.utils.data import DataLoader
from make_csv import get_random
import time
path = "../data/market/Market-1501-v15.09.15/"
lr = 0.001
batch = 16
subepochs = 3


model = Model()

model.to("cuda"  if torch.cuda.is_available() else "cpu")

criterion  = nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)
best = np.Inf
last_saved = None
thresh = 5
model_path  = "./model/"
epochs = 20
model_name = "reidmodel_final_last.pt"
length =4000
total_start_time = time.time()
paths = ["/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/dukemtmc/bounding_box_train/","/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/market/Market-1501-v15.09.15/bounding_box_train/","/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/PKU-Reid-Dataset/PKUv1a_128x48/"]
for epoch in tqdm(range(epochs),desc = "Epochs"):
    print("                                   Starting Epoch: ",epoch+1,"\n")
    df = pd.DataFrame(columns=["Anchor","Positive","Negative","Dataset","Path"])
    epoch_start_time = time.time()
    for X in tqdm(paths,desc = "Datasets"):
        na = X.split("/")[-3].split("-")[0]
        for image in tqdm(range(length),total=length,desc = f" Loading {na} Images"):
            anchor,positive,negative = get_random(X)
            df = df._append({"Anchor":anchor,"Positive":positive,"Negative":negative,"Dataset":na,"Path":X},ignore_index=True)

    trainset,valset = train_test_split(df,shuffle=True,random_state=42,test_size=0.2)
    trainset = Load_Data(trainset)
    valset = Load_Data(valset)
    trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
    valloader = DataLoader(valset,batch_size=batch,shuffle=True)
    for subepoch in tqdm(range(subepochs),desc = "SubEpochs"):

        train_loss = train_model(model,trainloader,criterion,optimizer,device = "cuda")

        val_loss = eval_model(model,valloader,criterion,device = "cuda")
        if val_loss < best:
            torch.save(model.state_dict(),model_path+model_name)
            best = val_loss
            print(f"*******************Best Model saved at Epoch {epoch+1}  and Subepochs at {subepoch+1} ******************\n")
            last_saved = epoch
        print(f"Epochs: {epoch+1} SubEpochs: {subepoch+1} Train loss: {train_loss} Val loss: {val_loss}\n")
    if last_saved and epoch-last_saved > thresh:
        print(f"Call Back is activated last saved at:  {last_saved}  AND CURRENT EPOCH IS : {epoch} \n")
        print(f"Epoch: {epoch+1} Takes Time:",round((time.time()-epoch_start_time)/60,2)," Minutes \n")
        break
    print(f"Epoch: {epoch+1} Takes Time:",round((time.time()-epoch_start_time)/60,2)," Minutes \n")
    # print(f"Epochs: {epoch+1} Train loss: {train_loss} Val loss: {val_loss}")
print("Total Training Time:",round((time.time()-total_start_time)/60,2)," Minutes")