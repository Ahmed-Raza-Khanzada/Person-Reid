from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2



def extract_features(image):
  model = InceptionResNetV2(include_top=False, weights='imagenet')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
  image = cv2.resize(image, (299, 299)) 
  x = img_to_array(image)
  x = np.expand_dims(x, axis=0)  
  x = preprocess_input(x)
  features = model.predict(x)
  return features


