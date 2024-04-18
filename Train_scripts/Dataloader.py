import torch as nn
from skimage import io
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
class Load_Data(Dataset):
    def __init__(self,df,path = "../data/market/Market-1501-v15.09.15/bounding_box_train/"):
        self.df = df
        self.path = path
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        row  =  self.df.iloc[index]
      
        anchor   = nn.from_numpy(resize(io.imread(row["Path"]+row["Anchor"]),(128,48)).astype(np.float32)).permute((2,0,1))/255.0
        positive = nn.from_numpy(resize(io.imread(row["Path"]+row["Positive"]),(128,48)).astype(np.float32)).permute((2,0,1))/255.0
        negative = nn.from_numpy(resize(io.imread(row["Path"]+row["Negative"]),(128,48)).astype(np.float32)).permute((2,0,1))/255.0
        return anchor,positive,negative
    

