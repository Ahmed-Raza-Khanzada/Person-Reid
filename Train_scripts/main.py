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
path = "../data/market/Market-1501-v15.09.15/"
lr = 0.001
batch = 16
epochs = 100
df  = pd.read_csv(path+"train.csv")

trainset,valset = train_test_split(df,shuffle=True,random_state=42,test_size=0.2)
model = Model()

model.to("cuda"  if torch.cuda.is_available() else "cpu")
trainset = Load_Data(trainset)
valset = Load_Data(valset)
trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
valloader = DataLoader(valset,batch_size=batch,shuffle=True)
criterion  = nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)
best = np.Inf
last_saved = None
thresh = 10
model_path  = "./model/"
for epoch in tqdm(range(epochs)):

    train_loss = train_model(model,trainloader,criterion,optimizer,device = "cuda")

    val_loss = eval_model(model,valloader,criterion,device = "cuda")
    if val_loss < best:
        torch.save(model.state_dict(),model_path+"reidmodel.pt")
        best = val_loss
        print(f"*******************Best Model saved at {epoch+1} ******************")
        last_saved = epoch
    if last_saved and epoch-last_saved > thresh:
        print(f"Call Back is activated last saved at:  {last_saved}  AND CURRENT EPOCH IS : {epoch} ")
        break
    print(f"Epochs: {epoch+1} Train loss: {train_loss} Val loss: {val_loss}")