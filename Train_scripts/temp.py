import torch as nn
from skimage import io
import os
from skimage.transform import resize
import numpy as np

for img in os.listdir("/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/PKU-Reid-Dataset/PKUv1a_128x48/"):
    try:
        nn.from_numpy(resize(io.imread("/home/osamakhan/Documents/Person_REID_OWNAPPROACH/data/PKU-Reid-Dataset/PKUv1a_128x48/"+img),(128,48)).astype(np.float32)).permute((2,0,1))/255.0
    except:
        print(img)