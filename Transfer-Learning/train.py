import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import DataLoader
from LSTM import get_model
from sklearn.metrics import f1_score
import os

BATCH_SIZE = 512
EPOCHS = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 1
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

datasets = ['tess', 'ravdess', 'savee', 'iemocap']
save_path = os.path.join( "checkpoints", "_".join(datasets) + '.pt' )

train_ds = DataLoader(path_csvs = [ os.path.join('../data', data, data + '_train.csv') for data in datasets ], 
feature_pkls = [ os.path.join('../data', data, data + '.pkl') for data in datasets ] )
val_ds = DataLoader(path_csvs = [ os.path.join('../data', data, data + '_val.csv') for data in datasets ], 
feature_pkls = [ os.path.join('../data', data, data + '.pkl') for data in datasets ] )
test_ds = DataLoader(path_csvs = [ os.path.join('../data', data, data + '_test.csv') for data in datasets ], 
feature_pkls = [ os.path.join('../data', data, data + '.pkl') for data in datasets ] )

train_dl = torch.utils.data.DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = True)

model, opt = get_model(ed = 20, hd = 128, ts = train_ds.num_emotions)
model.to(device)

'''
model_dict = "checkpoints/tess_ravdess_savee_iemocap.pt"
model.load_state_dict( torch.load(model_dict) )
for i, module in enumerate(model.modules()):
    if i == 1 or i == 2:
        for param in module.parameters():
            param.requires_grad = False
'''

loss_func = nn.NLLLoss()

def accuracy(outputs, labels):
    outputs = np.array( outputs )
    labels = np.array( labels )
    return np.sum(outputs == labels) / float(labels.size)

print ("Training Started...")
best_test_loss, best_test_acc, best_f1_score = None, None, None

for epoch in range(EPOCHS):

    model.train()
    train_loss_epoch, train_preds_epoch, train_lables_epoch =  [], [], []

    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = loss_func(out, yb)
        _, preds = torch.max(out, 1)

        loss.backward()
        opt.step()
        opt.zero_grad()

        loss = loss.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        lbls = yb.detach().cpu().numpy()

        train_loss_epoch.append(loss)
        train_preds_epoch.extend( list(preds) )
        train_lables_epoch.extend( list(lbls) )
   
    epoch_train_loss = np.mean(train_loss_epoch)
    epoch_train_acc = accuracy(train_preds_epoch, train_lables_epoch)

    model.eval()

    with torch.no_grad():
        test_loss_epoch, test_preds_epoch, test_labels_epoch =  [], [], []
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_func(out, yb)
            _, preds = torch.max(out, 1)

            loss = loss.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            lbls = yb.detach().cpu().numpy()

            test_loss_epoch.append( loss )
            test_preds_epoch.extend( preds )
            test_labels_epoch.extend( lbls )

    epoch_test_loss = np.mean(test_loss_epoch)
    epoch_test_acc = accuracy( test_preds_epoch, test_labels_epoch )
    epoch_f1_score = f1_score(test_labels_epoch, test_preds_epoch, average='weighted')

    if best_test_acc is None or best_test_acc < epoch_test_acc:
        best_test_loss = epoch_test_loss
        best_test_acc = epoch_test_acc
        best_f1_score = epoch_f1_score
        torch.save(model.state_dict(), save_path)


    print("-"*50)
    print("Epoch :", epoch )
    print ("Training Loss :", epoch_train_loss)
    print ("Training Acc :", epoch_train_acc)
    print ("Testing Loss :", epoch_test_loss)
    print ("Testing Acc :", epoch_test_acc)
    print ("Best Test Acc :", best_test_acc)
    print ("Best F1 Score :", best_f1_score)
    print ("-"*50)
