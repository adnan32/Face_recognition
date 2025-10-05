# Local Face Recognition using InsightFace & OpenCV

This project implements a local, real-time face recognition system using InsightFace (RetinaFace + ArcFace) and OpenCV.
It builds a gallery of embeddings from your own dataset and performs live recognition via webcam all running offline.

## Dataset

You can use your own images or download a public dataset such as:
 [Pins Face Recognition Dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition/data)

Each person should have their own folder:

```
dataset/
  pins_Adnan/
    1.jpg
    2.jpg
  pins_Karim/
    1.jpg
    2.jpg
```

## Installation

Create and activate a virtual environment (optional but recommended).
Then install the required dependencies:

```bash
python -m pip install "numpy<2.0" "opencv-python<4.12" onnx==1.14.1 onnxruntime==1.17.3 insightface==0.7.3 matplotlib
```

All versions used are also listed in `requirements.txt`.

##  Verify Dependencies

Before proceeding, check that everything works correctly:

```python
import cv2
import numpy as np
from insightface.app import FaceAnalysis
print("ok")
```

Then check your available camera index:

```python
import cv2
for i in range(3):
    cap = cv2.VideoCapture(i)
    print(i, cap.isOpened())
    cap.release()
```

## Project Workflow

### 1. Experiment in Jupyter Notebook
Start by experimenting with one image to understand the detection and embedding workflow using FaceAnalysis.

Example:

```python
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(1536,1536), det_thresh=0.14)
faces = app.get(img)
e = faces[0].embedding
print("embedding shape:", e.shape, "norm:", np.linalg.norm(e))
```

You can visualize detections and crops using matplotlib.

### 2. Embed All Images (Build the Gallery)
Run the embedding script to process all images in your dataset and create a .npz file that stores names and mean embeddings per person.

```bash
python embedding_all_images.py
```

Each person’s embeddings are averaged into a 512-D normalized vector and saved as:

```
gallery_embeddings.npz
├── names  → ['Adnan', 'Karim', ...]
└── means  → [[0.55, 0.83, ...], [0.60, 0.79, ...], ...]
```

### 3. Run Live Face Recognition
The recognition script opens your webcam, detects faces, computes embeddings, and compares them against your gallery using cosine similarity.
If similarity > threshold (default 0.35), the system displays the recognized name; otherwise, it shows “Unknown”.

```bash
python recognition_webcam.py
```

## How It Works

- **Detection:** RetinaFace finds and aligns faces.
- **Embedding:** ArcFace converts each face into a 512-D vector (identity features).
- **Normalization:** Each vector is normalized to unit length (L2 norm = 1).
- **Comparison:** Cosine similarity measures closeness between embeddings.
- **Thresholding:** If similarity > 0.35 → recognized, else Unknown.

##  Example Output

**Embedding stage:**

```
person:  Adnan
pins_Adnan: 33 samples, mean norm=1.000
```
<img width="353" height="334" alt="Skärmbild 2025-10-06 013401" src="https://github.com/user-attachments/assets/929a4366-95f4-4238-add0-d530027f0af0" />

**Recognition stage:**
Live webcam feed shows bounding boxes and labels like:

```
Adnan  0.73
Unknown  0.18
```

## Tech Stack

- Python 3.10+
- InsightFace 0.7.3 (RetinaFace + ArcFace models)
- OpenCV for image and video handling
- NumPy for embedding operations
- Matplotlib for visualization

##  Common Challenges

- Library version compatibility (especially numpy, onnx, and onnxruntime).
- Small, cropped faces may fail detection → upscale & pad images to ~1536 pixels.
- On some systems, the GPU (CUDA) provider may not be available, CPU works fine.


## ✍️ Author

**Adnan Altukleh**  
AI & Machine Learning Engineer  
📍 Sweden  
📧 [adnantakleh12@gmail.com](mailto:adnantakleh12@gmail.com)
