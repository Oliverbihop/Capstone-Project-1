
# import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
from model_VGG16 import VGG16
import torch
import matplotlib.pyplot as plt
import os 
import torch.nn.functional as F
import time
from torchvision import transforms
import torch.nn as nn
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


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 2, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(512, 6) # stride 1: 2304, 2:512
        #self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x) #48.48.32     #32.32.8
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x) #24.24.32     #16.16.16
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x) #12.12.64     #8.8.32
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2) #6.6.64  #4.4.32
        # x = self.dropout1(x)
        x = torch.flatten(x, 1) # 2304
        x = self.dropout1(x)
        #x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.leaky_relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
classes = ['uyen', 'nien', 'tri', 'tin', 'khoa','none']

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#print('device:', device)

# Define hyper-parameter
img_size = (64, 64)

# define model
model = Network()
my_model = model.load_state_dict(torch.load('./cnn1.pt', map_location=torch.device('cpu')))

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
        loader = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
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
           