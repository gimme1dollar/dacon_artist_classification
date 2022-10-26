import random
import pandas as pd
import numpy as np
import os
import cv2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

from .utils import *

import warnings
warnings.filterwarnings(action='ignore') 


def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values



def train(model, optimizer, train_loader, test_loader, scheduler, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1,CFG["EPOCHS"]+1):
        model.train()
        train_loss = []
        for img, label in tqdm(iter(train_loader)):
            img, label = img.float().to(device), label.to(device)
            
            optimizer.zero_grad()

            model_pred = model(img)
            
            loss = criterion(model_pred, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)
            
        val_loss, val_score = validation(model, criterion, test_loader, device)
            
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_score < val_score:
            best_model = model
            best_score = val_score

#             torch.save({'epoch':epoch,
#                         'model_state_dict':model.state_dict(), 
#                         'optimizer_state_dict':optimizer.state_dict(),
#                         'loss':val_loss,
#                         },
#                         f"/content/drive/MyDrive/eff/1023_{epoch}_{val_score:.3f}.pt")
            
        
    return best_model


def validation(model, criterion, test_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.to(device)
            
            model_pred = model(img)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1



def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    
    model_preds = []
    
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)
            
            model_pred = model(img)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    
    print('Done.')
    return model_preds



if __name__ == "__main__":
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    CFG = {
    'IMG_SIZE':224,
    'EPOCHS':20,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':64,
    'SEED':41
    }

    seed_everything(CFG['SEED']) # Seed 고정


    df = pd.read_csv('train.csv')
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)

    train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=CFG['SEED'])

    train_df = train_df.sort_values(by=['id'])

    val_df = val_df.sort_values(by=['id'])

    train_img_paths, train_labels = get_data(train_df)
    val_img_paths, val_labels = get_data(val_df)

    train_transform = A.Compose([
    A.Resize(480,480), # 299
    A.RandomCrop(224,224),
    A.HorizontalFlip(p = 0.5), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
    ])

    test_transform = A.Compose([
    A.Resize(480, 480),
    A.RandomCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
    ])


    train_dataset = CustomDataset(train_img_paths, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val_img_paths, val_labels, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = BaseModel()
    model.eval()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = None 

    infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

    test_df = pd.read_csv('test.csv')

    test_img_paths = get_data(test_df, infer=True)

    test_dataset = CustomDataset(test_img_paths, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


    preds = inference(infer_model, test_loader, device)



    preds = le.inverse_transform(preds) # LabelEncoder로 변환 된 Label을 다시 화가이름으로 변환

    submit = pd.read_csv('sample_submission.csv')

    submit['artist'] = preds

    submit.to_csv('sample/sample_1026.csv', index = False)