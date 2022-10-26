import cv2
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

from utils import *
from model import *
from dataset import *
from inference import *

import warnings
warnings.filterwarnings(action='ignore') 



def train(model, optimizer, train_loader, test_loader, scheduler, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1,args.epoch+1):
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

            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(), 
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':val_loss,
                        },
                        f"saved/eff_{epoch}_{val_score:.3f}.pt")
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    seed_everything(args.seed)

    # Dataset
    train_dataset, val_dataset = get_dataset()
    train_loader = DataLoader(train_dataset, batch_size =args.batch_size,shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Train model
    model = BaseModel(num_classes=50)
    model.eval()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate)
    scheduler = None 

    trained_model = train(model, optimizer, train_loader, val_loader, scheduler, args.device)

    # Inference models
    test_df = pd.read_csv('test.csv')
    test_img_paths = get_data(test_df, infer=True)

    test_dataset = CustomDataset(test_img_paths, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Submit
    preds = inference(trained_model, test_loader, args.device)
    preds = le.inverse_transform(preds) # LabelEncoder로 변환 된 Label을 다시 화가이름으로 변환

    submit = pd.read_csv('sample_submission.csv')
    submit['artist'] = preds
    submit.to_csv('sample/sample_1026.csv', index = False)
