import sklearn
import joblib
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time



from playsound import playsound
import warnings
warnings.filterwarnings("ignore")


lb = joblib.load('lb.pkl')
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # changed 3 to 1
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(lb.classes_))
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


print('Loading model and label binarizer...')
lb = joblib.load('lb.pkl')
model = CustomCNN()
print('Model Loaded...')
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
print('Loaded model state_dict...')
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])


class detection:
  countDrowning=0
  ThersholdForDrowning =  30;	
  fram=0
  def detectDrowning(self):
    #input from the camera
    cap = cv2.VideoCapture("videos/notdrowning.mp4")
    if (cap.isOpened() == False):
        print('Error while trying to read video')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    while(cap.isOpened()):
        start_time = time.time()
        ret, frame = cap.read()
        if ret == True:
            model.eval()
            with torch.no_grad():
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_image = aug(image=np.array(pil_image))['image']
                if self.fram == 500:
                 break
                self.fram+=1
                pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
                pil_image = pil_image.unsqueeze(0)
                outputs = model(pil_image)
                _, preds = torch.max(outputs.data, 1)
            #self.calculateDrowning(lb.classes_[preds])
            print("Frame classified as: ",lb.classes_[preds])
        else: 
            break
            

        print("FPS: ", int(1 / (time.time() - start_time))) 

    #print(self.countDrowning)
    return 0
    


d = detection()
d.detectDrowning()



