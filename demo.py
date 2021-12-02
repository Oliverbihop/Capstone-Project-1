
# import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
from model_VGG16 import VGG16
import torch
import matplotlib.pyplot as plt
import os 
import time
from torchvision import transforms
from PIL import Image, ImageOps
from torch.autograd import Variable
#stops = cv2.CascadeClassifier('haar_cascade/stop_sign.xml')
#rights = cv2.CascadeClassifier('haar_cascade/Right.xml')
#aheads = cv2.CascadeClassifier('haar_cascade/Ahead.xml')
# path = "3.png"
# img_read = cv2.imread(path)
foder = 'none'   #ahead: 14, noright: 12, none: 124, 

#path = './dataset/test_data/'+str(foder) #stop -> right, ahead-> stop, right -> stop
#test_img = os.listdir(path)
# print(test_img)

# def output(frame, pre):
#     print("boundingbox: ", pre)
#     y1 = pre[0][1]
#     y2 = pre[0][1] + pre[0][2]
#     x1 = pre[0][0]
#     x2 = pre[0][0] + pre[0][3]
#     img = frame[y1:y2, x1:x2]
#     # img = frame[17:17+43, 113:113+43]
#     # cv2.imshow("asd", img)
#     # cv2.waitKey(0)
#     return img


# def detect_TrafficSign(frame):
#     # frame=frame[40:180,160:320]
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     stop_pre = stops.detectMultiScale(gray, #X,Y - W,H
#     scaleFactor=1.05,
#     minNeighbors=2,
#     minSize=(20, 20),
#     maxSize=(60, 60),
#     flags = 0)
#     right_pre = rights.detectMultiScale(gray, #X,Y - W,H
#     scaleFactor=1.05,
#     minNeighbors=2,
#     minSize=(20, 20),
#     maxSize=(60, 60),
#     flags = 0)
#     ahead_pre = aheads.detectMultiScale(gray, #X,Y - W,H
#     scaleFactor=1.05,
#     minNeighbors=2,
#     minSize=(20, 20),
#     maxSize=(60, 60),
#     flags = 0)
    
#     if len(stop_pre)>0:
#         print("stops")
#         img = output(frame, stop_pre)
#         Run_Predict(img)

#     if len(right_pre)>0:
#         print("rights")
#         img = output(frame, right_pre)
#         Run_Predict(img)

#     if len(ahead_pre)>0:
#         print("ahead")
#         img = output(frame, ahead_pre)
#         Run_Predict(img)

    
classes = ['uyen', 'thayphuc', 'phuoc', 'nam', 'khoa']

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#print('device:', device)

# Define hyper-parameter
img_size = (48, 48)

# define model
model = VGG16()
my_model = model.load_state_dict(torch.load('./epoch_39.pth', map_location=torch.device('cpu')))

#port to model to gpu if you have gpu
model = model.to(device)
model.eval()
def Predict(img_raw):
        # resize img to 48x48
        
        #global device, model, img_size
        #img_rgb = cv2.resize(img_raw, img_size)
        #img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # normalize img from [0, 255] to [0, 1]
        #img_rgb = img_rgb/255
       # img_rgb = img_rgb.astype('float32')
        #img_rgb = img_rgb.transpose(2,0,1)
        #img1 = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
     
        img1 = Image.fromarray(img_raw)
        loader = transforms.Compose([transforms.Resize((48, 48)),transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        img_tensor = loader(img1).unsqueeze(0)
        img_rgb = Variable(img_tensor, requires_grad=False)


        # convert image to torch with size (1, 1, 48, 48)
        #img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

        with torch.no_grad():
            img_rgb = img_rgb.to(device)
            # print("type: " + str(numb), type(img_rgb))
            y_pred = model(img_rgb)
            # print("y_pred", y_pred)           
            _, pred = torch.max(y_pred, 1)
            
            pred = pred.data.cpu().numpy()
            # print("2nd", second_time - fist_time)
            # print("predict: " +str(numb), pred)
            class_pred = classes[pred[0]]
            # print("class_pred", class_pred)
            
            
        return class_pred
           