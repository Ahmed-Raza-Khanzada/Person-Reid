from skimage import io
from model import Model
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_closest_imgs


def euclidean_dist(img_enc, anc_enc_arr):
    dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T))
    return dist
DEVICE = "cuda"
df_enc = pd.read_csv("../data/mydata_embbedings.csv")
idx = 605
DATA_DIR = "/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data"
img_name = df_enc["images"].iloc[idx]
img_path = DATA_DIR + img_name
print("**********????????***********",img_path)
img = io.imread(img_path)
img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
model = Model()
model.to("cuda")
model.load_state_dict(torch.load("model/reidmodel.pt"))

model.eval()
with torch.no_grad():
    img = img.to(DEVICE)
    img_enc = model(img.unsqueeze(0))
    img_enc = img_enc.detach().cpu().numpy()

anc_enc_arr = df_enc.iloc[:, 1:].to_numpy()
anc_img_names = df_enc["images"]

distance = []

for i in range(anc_enc_arr.shape[0]):
    dist = euclidean_dist(img_enc, anc_enc_arr[i : i+1, :])
    distance = np.append(distance, dist)
closest_idx = np.argsort(distance)
img_path = "/".join(img_path.split("/")[-2:])
DATA_DIR +="/" 
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^6")
# plt.imshow(img.cpu().reshape((128, 64, 3)))
plot_closest_imgs(anc_img_names, DATA_DIR, img, img_path, closest_idx, distance, no_of_closest = 50)