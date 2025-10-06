import cv2, numpy as np
from insightface.app import FaceAnalysis

import warnings
warnings.filterwarnings("ignore", message=".*rcond.*lstsq.*")


data=np.load("gallery_embeddings.npz",allow_pickle=True)

NAMES=list(data["names"])
MEANS=data["means"]

def cos(a,b):
    return float(np.dot(a,b))


app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

THRESH = 0.35  
CAM_INDEX = 0 

cap=cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

while True:
    ok,frame=cap.read()
    
    if not ok:
        break
    
    faces=app.get(frame)
    for f in faces:
        x1,y1,x2,y2=f.bbox.astype(int)
        e=f.embedding.astype("float32").reshape(-1)
        e/=(np.linalg.norm(e)+1e-8)


        similarty_search=np.array([cos(e,m) for m in MEANS])
        best_match_ind = int(np.argmax(similarty_search))
        best_name=NAMES[best_match_ind]
        best_sim = similarty_search[best_match_ind]
        label = best_name if best_sim > THRESH else "Unknown"
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 2)
        cv2.putText(frame, f"{label} {best_sim:.2f}", (x1, max(0,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # Esc or q to quit
        break
cap.release()
cv2.destroyAllWindows()