import cv2
import matplotlib.pyplot as plt
from .superres import SuperRes

class SuperRes:
    def __init__(self) -> int:
        self.version = 1.0
    
    def run(model: str) -> None:
        img = cv2.imread(model)
        
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel("video2x/models/LapSRN_x8.pb")
        sr.setModel("lapsrn", 8)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        result = sr.upsample(img)
        return result

def test_opencv_superres():
    model = "data/test_image.png"
    img = cv2.imread(model)
    dims = img.shape[1]
    sr = SuperRes()
    assert sr.run(model) ==  dims * 8
