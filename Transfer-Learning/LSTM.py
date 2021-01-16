import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM_emorec(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size):
        super(LSTM_emorec, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_1 = nn.LSTM(embedding_dim, hidden_dim , batch_first = True)
        self.lstm_2 = nn.LSTM(hidden_dim, hidden_dim , batch_first = True)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2targ = nn.Linear(hidden_dim, target_size)

    def forward(self,x):
        lstm_out ,_ = self.lstm_1(x)
        _ ,(lstm_out, _) = self.lstm_2(lstm_out)
        lstm_out = lstm_out[-1]
        hidden_output = self.hidden(lstm_out)
        #hidden_output = F.relu(hidden_output)
        out = self.hidden2targ(hidden_output)
        out = F.log_softmax(out, dim=1)
        return out

def get_model(ed, hd, ts):
    model = LSTM_emorec(embedding_dim = ed, hidden_dim = hd, target_size = ts)
    opt = optim.Adam( model.parameters(), lr=0.0002 )
    return model, opt

if __name__ == '__main__':
    model, opt = get_model(20, 128, 5)
    Y = model( torch.zeros( (2, 120, 20) ) )
    print ( Y.shape )