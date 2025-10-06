import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*rcond.*lstsq.*")

app=FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(1536,1536), det_thresh=0.14)


root="105_classes_dataset"
names,means=[],[]


def embadding_image(img, app):
    faces = app.get(img)
    if not faces:
        return None
    e = faces[0].embedding.astype("float32")
    e /= (np.linalg.norm(e) + 1e-8) # Vector normilzing length to 1
    return e #(512-D embedding vector)

def img_rescal(img):
    h, w = img.shape[:2]
    scale = 1536.0 / max(h, w)
    if scale > 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    pad = int(max(img.shape[:2]) * 0.80)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT,value=(255, 255, 255))
    return img

for person in sorted(os.listdir(root)):

    pdir = os.path.join(root, person)
    if not os.path.isdir(pdir): 
        continue
    print("person: ",person[5:]," \n")
    emb=[]
    count=1
    for path_image in os.listdir(pdir):
        print("Prosecsing the image: ",count,)
        image_path=os.path.join(pdir,path_image)
        img=cv2.imread(image_path)
        if img is None: 
            continue
        scaled_img=img_rescal(img)
        embs=embadding_image(scaled_img,app)

        
        if embs is not None and len(embs) > 0:
            emb.append(embs)
        count+=1
    if len(emb) == 0:
        print(f"[skip] {person}: no usable faces")
        continue

    E = np.stack(emb)          # (N, 512)
    mean = E.mean(axis=0)           # (512,)
    mean /= (np.linalg.norm(mean) + 1e-8)

    names.append(person[5:])        # keep if you really want to strip a prefix
    means.append(mean)
    print(f"{person}: {E.shape[0]} samples, mean norm={np.linalg.norm(mean):.3f}")
np.savez("gallery_embeddings.npz", names=np.array(names, dtype=object), means=np.stack(means))

