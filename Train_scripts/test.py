from model import Model
import torch
from skimage import io
import pandas as pd 
from tqdm import tqdm
import numpy as np
path = "../data/"

df  = pd.read_csv(path+"my_data.csv")
datadir ="../data/market/Market-1501-v15.09.15/bounding_box_train/"

def make_prediction(names,model,datapath = datadir,usedir = True):
    model.eval()
    encodings = []
    with torch.no_grad():
        for image in tqdm(np.array(names)):
            if usedir:
                image = torch.from_numpy(io.imread(datapath + image)).permute((2,0,1))/255.0
            else:
                image = torch.from_numpy(io.imread("."+image)).permute((2,0,1))/255.0

            image = image.to("cuda")
            rawembeddind = model(image.unsqueeze(0))
            encodings.append(rawembeddind.squeeze().cpu().detach().numpy())
    encodings = np.array(encodings)
    enc = pd.DataFrame(encodings)
    names = df["images"].apply(lambda x :"/"+"/".join( x.split("/")[-2:]))

    df_enc = pd.concat([names,enc],axis = 1)
    return df_enc
model = Model()
model.to("cuda")
model.load_state_dict(torch.load("model/reidmodel.pt"))

final = make_prediction(df["images"],model,usedir=False)
print(final.head())
final.to_csv(f"../data/mydata_embbedings.csv",index = False)