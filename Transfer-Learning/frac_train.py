import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from os.path import join
from torch.utils.data import DataLoader
from datasets.shemo_files.shemo_dataloader import DataLoader as Dataset
from sklearn.metrics import f1_score
from LSTM import LSTM_emorec

LR = 1e-6
MAX = 2e-5
BATCH_SIZE = 64
EPOCHS = 2500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 1
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

loss_func = nn.NLLLoss()

dataset_name = 'shemo'
root = join('datasets', f'{dataset_name}_files')
f = open(f'{dataset_name}.txt', 'a+')


def accuracy(outputs, labels):
    outputs = np.array(outputs)
    labels = np.array(labels)
    return np.sum(outputs == labels) / float(labels.size)


for model_path in [None, 'tess.pt', 'emodb.pt', 'ravdess.pt', 'iemocap.pt', 'savee.pt']:
    if model_path is not None:
        model_dict = torch.load(model_path)

        del model_dict['hidden.weight']
        del model_dict['hidden.bias']
        del model_dict['hidden2targ.weight']
        del model_dict['hidden2targ.bias']

        f.write(model_path + '\n')

    for FRACTION in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        train_ds = Dataset(path_csv=join(root, f'{dataset_name}_train.csv'),
                           feature_pkl=join(root, f'{dataset_name}.pkl'), fraction=FRACTION)
        test_ds = Dataset(path_csv=join(root, f'{dataset_name}_test.csv'),
                          feature_pkl=join(root, f'{dataset_name}.pkl'), fraction=1)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = LSTM_emorec(embedding_dim=20, hidden_dim=128, target_size=train_ds.num_emotions)
        opt = optim.Adam(model.parameters(), LR)

        sched = optim.lr_scheduler.CyclicLR(opt, LR, MAX, cycle_momentum=False, mode='triangular2')

        if model_path is not None:
            model.load_state_dict(model_dict, strict=False)
            for i, module in enumerate(model.modules()):
                if i == 1 or i == 2:
                    for param in module.parameters():
                        param.requires_grad = False

        model.to(device)

        print("Training Started...")
        best_test_loss, best_test_acc, best_f1_score = None, None, None

        for epoch in range(EPOCHS):

            model.train()
            train_loss_epoch, train_preds_epoch, train_lables_epoch = [], [], []

            for i, (xb, yb) in enumerate(train_dl):
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = loss_func(out, yb)
                _, preds = torch.max(out, 1)

                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()

                loss = loss.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                lbls = yb.detach().cpu().numpy()

                train_loss_epoch.append(loss)
                train_preds_epoch.extend(list(preds))
                train_lables_epoch.extend(list(lbls))

            epoch_train_loss = np.mean(train_loss_epoch)
            epoch_train_acc = accuracy(train_preds_epoch, train_lables_epoch)

            model.eval()

            with torch.no_grad():
                # valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
                test_loss_epoch, test_preds_epoch, test_labels_epoch = [], [], []
                for xb, yb in test_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = loss_func(out, yb)
                    _, preds = torch.max(out, 1)

                    loss = loss.detach().cpu().numpy()
                    preds = preds.detach().cpu().numpy()
                    lbls = yb.detach().cpu().numpy()

                    test_loss_epoch.append(loss)
                    test_preds_epoch.extend(preds)
                    test_labels_epoch.extend(lbls)

            epoch_test_loss = np.mean(test_loss_epoch)
            epoch_test_acc = accuracy(test_preds_epoch, test_labels_epoch)
            epoch_f1_score = f1_score(test_labels_epoch, test_preds_epoch, average='weighted')

            if best_test_acc is None or best_test_acc < epoch_test_acc:
                best_test_loss = epoch_test_loss
                best_test_acc = epoch_test_acc
                best_f1_score = epoch_f1_score

            print("-" * 50)
            print("Epoch :", epoch)
            print("Training Loss :", epoch_train_loss)
            print("Training Acc :", epoch_train_acc)
            print("Testing Loss :", epoch_test_loss)
            print("Testing Acc :", epoch_test_acc)
            print("Best Test Acc :", best_test_acc)
            print("Best F1 Score :", best_f1_score)
            print("-" * 50)

        f.write(str(best_f1_score) + '\n')

f.close()