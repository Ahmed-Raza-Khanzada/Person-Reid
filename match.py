import torch
import numpy as np
import cv2

def euclidean_dist(img_enc, anc_enc_arr):
    dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T))
    return dist


def make_embedding(model,img,DEVICE = "cuda"):

    # img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
    img = torch.from_numpy(cv2.resize(img, (48,128)).astype(np.float32)).permute(2, 0, 1) / 255.0
 

    model.eval()
    with torch.no_grad():
        img = img.to(DEVICE)
        img_enc = model(img.unsqueeze(0))
        img_enc = img_enc.detach().cpu().numpy()
        
    return img_enc


def make_matching(em1,em2):
    # em1 = make_embedding(model,img)
    # em2 = make_embedding(model,img2)
    # print("Matching")
    return euclidean_dist(em1,em2)
