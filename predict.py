from ultralytics import YOLO
import cv2 
import numpy as np
import gc
import os
from matplotlib.cm import get_cmap
import time

from features_extractor import extract_features
from features_match import feature_score, ssim_score
from skimage.metrics import structural_similarity as ssim


cmap = get_cmap('tab10')
def draw_bbox(image, x, y, x2,y2, color):
    frame = image.copy()
    box_coords = [(x, y), (x2,y), (x2, y2), (x, y2)]
    overlay = image.copy()
    transparent_box_alpha = 0.4
    cv2.fillPoly(overlay, [np.array(box_coords, np.int32)], color)
    cv2.addWeighted(overlay, transparent_box_alpha, image, 1, 0, image)
    frame[y:y2,x:x2] = image[y:y2,x:x2]
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    # image[y:y2,x:x2] = overlay
    return frame


def get_features(image,x1,y1,x2,y2):
    image = image[y1:y2,x1:x2]
    features = extract_features(image)
    return features

features_vector = {}
thresh_features = 10
thresh_frame = 5

def calc_score(matrix1, matrix2):
   s1 = feature_score(matrix1, matrix2)
   s2 = ssim_score(matrix1, matrix2)
   return s1+s2/2

def match_images(new_feature, old_images_features, calc_score=calc_score):

    distance_matrix = np.empty(len(old_images_features))
    for j,(key, old_features_list) in enumerate(old_images_features.items()):
        for old_feature in old_features_list[1]:
            distance_matrix[ j] = calc_score(new_feature, old_feature)

#   matched_ids = np.argmin(distance_matrix, axis=1)
    return distance_matrix
# np.map(lambda y: my_func(element, y), arr2)

final = {}
l = [[1,9],[6,8,11],[4,7],[2],[3]]

def load_array(path):
    d= {}
    for pos,np_array in enumerate(os.listdir(path)):
        d[pos] = np.load(path+"/"+np_array)
    return d


def infer(path,f, model="Nano", u=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']):

    ids = set()

    model = YOLO("yolov8x.pt")
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    kwargs = {'device': 0, 'verbose': False,"conf":0.20,"classes":0}

   
    n = 0

    for results in model.track(source=path,persist=True,stream=True, **kwargs):
        frame = results.orig_img
        # print("Reults: ",results.boxes.cpu().numpy())
        # print(results)


        for result in results.boxes.cpu().numpy():
            if result.id:
                x1, y1, x2, y2 = result.xyxy.astype(int).tolist()[0]
                track_id = result.id[0].astype(int)
                track_id = str(track_id)
                feature = get_features(frame,x1,y1,x2,y2) 
                if track_id  not in features_vector:
                    f_time = time.time()
                    features_vector[track_id] = (n,[feature])
                    print("***************",track_id,n)
                    print("Time to get features: ",time.time()-f_time)                       
                else:
               
                    last = features_vector[track_id][1]
                    f_time = time.time()
                    if ((n-features_vector[track_id][0])>thresh_frame) and len(last)<thresh_features:
                        last.append(feature)
                        features_vector[track_id] = (n,last)
                        print("###########################",track_id,n)
                        print("Time to get features: ",time.time()-f_time)

            ########################################Remove this piece of code###########################################################
                turn = False
                for l1 in l:
                    if int(track_id) in l1:
                        for ele1 in l1:
                            if  ele1 in final.keys():
                                print("Both Same: ",track_id,ele1)
                                ele = final[ele1]
                                ele.append(feature)
                                final[int(track_id)] = ele
                            else:
                                print("New",track_id)
                                final[int(track_id)] = [feature]
                            turn = True
                            break
                    if turn:
                        break
            ##################################################################################################
            ##############################################################################################################################33
                cls = round(result.cls.item())
                color = (np.array(cmap(cls))*255)[-2::-1].astype(np.uint8).tolist()
                frame = draw_bbox(frame,x1,y1,x2,y2,color)
                cv2.putText(frame, u[cls]+" "+str(track_id), (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 2, cv2.LINE_AA)
                
        n += 1
 
        m = n//(60*f)
        s = n%(60*f)//f + round(n%(60*f)%f/f, 3)

        yield frame #{'frame':frame, 'time': f'{int(m)}m {s}s', 'count': count}

   
    del model
    gc.collect() 
t_time = time.time()
vid = "1099731453-preview.mp4"
start_time = time.time()
if vid.endswith(".mp4"):
    cap = cv2.VideoCapture(vid)
    h,w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    w,h = int(w),int(h)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    wrt2 = cv2.VideoWriter("./predict_custom_all_extra_final.mp4", cv2.VideoWriter.fourcc(*"mp4v"), 60, (w, h))
    for frame in infer(f"./{vid}",fps):
        # cv2.imshow("Frame",frame)
        # cv2.waitKey(1)
        wrt2.write(frame)
    print(vid ,"Done")
    print(f"--- {round((time.time() - start_time)/60,2)} minutes ---" )
print(f"Total Time: {round((time.time() - t_time)/60,2)}")
for i in final.keys():
    np.save("data/"+str(i)+".npy",final[i])
    

cv2.destroyAllWindows()