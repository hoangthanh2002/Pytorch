import glob 
import os.path as osp 
import random 
import numpy as np 
import json 
from PIL import Image
import matplotlib.pyplot as plt 
#%matplotlib inline

import torch
import torch.nn as nn 
import torch.optim as optim # import optimizer như adam, sgd
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from tqdm import tqdm

torch.manual_seed(1234) # cố định random seed để mỗi lần chạy code sẽ cho ra kết quả giống nhau
np.random.seed(1234)
random.seed(1234)

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225) 

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])    
            }
    def __call__(self, img, phase ='train'):
        return self.data_transform[phase](img)

def make_datapath_list(phase='train'):
    rootpath = './data/hymenoptera_data/'
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

path_list = make_datapath_list ('train')

train_img_list = make_datapath_list(phase='train')
val_img_list = make_datapath_list(phase='val')

#dataset
class Mydataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        
        if self.phase == 'train' :
            label = img_path[30:34]
        
        elif self.phase == 'val':
            label = img_path[28:32]
        if label == 'ants':
            label = 1
        if label == 'bees':
            label = 0
        return img_transformed, label

train_dataset = Mydataset(file_list=train_img_list, transform =ImageTransform(size, mean, std), phase='train') 
val_dataset = Mydataset(file_list=val_img_list, transform =ImageTransform(size, mean, std), phase='val')

#dataloader
batch_size = 4
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloader_dict = { "train" : train_dataloader , "val" : val_dataloader}

#network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
print(net)

#setting mode train
#net =net.train()  
criterior = nn.CrossEntropyLoss()

params_to_update = []
update_param_names = ["classifier.6.weight", "classifier.6.bias"]
for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True # chấp nhận cập nhật tham số
        params_to_update.append(param)
        print(name)
    else :
        param.requires_grad = False # không chấp nhận cập nhật tham số
print(params_to_update)
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9) #lr là learning rate 

def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()# eval là chế độ đánh giá, không cập nhật trọng số
            
            epoch_loss = 0.0 
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == 'train'):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):  
                    outputs = net (inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

num_epochs = 2
train_model(net, dataloader_dict, criterior, optimizer, num_epochs=num_epochs)