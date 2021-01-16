import torch.nn as nn
import torch.nn.functional as F





# class _Encoder(nn.Module):
#     def __init__(self, layers):
#         super(_Encoder, self).__init__()
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.layers(x)
#         x = x.view(x.size(0), -1)

#         return x

class _Encoder(nn.Module):
    def __init__(self, model):
        super(_Encoder, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        #x = x.view(x.size(0), -1)
        return x


class _Decoder(nn.Module):
    def __init__(self, output_size):
        super(_Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.layers(x)

        return x


class _Model(nn.Module):
    def __init__(self, output_size, encoder):
        super(_Model, self).__init__()
        self.encoder = encoder
        self.decoder = _Decoder(output_size=output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class LSTM_emorec(nn.Module):
    def __init__(self, embedding_dim = 20, hidden_dim = 128):
        super(LSTM_emorec, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_1 = nn.LSTM(embedding_dim, hidden_dim , batch_first = True)
        self.lstm_2 = nn.LSTM(hidden_dim, hidden_dim , batch_first = True)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,x):
        lstm_out ,_ = self.lstm_1(x)
        _ ,(lstm_out, _) = self.lstm_2(lstm_out)
        lstm_out = lstm_out[-1]
        hidden_output = self.hidden(lstm_out)
        return hidden_output


def Model(num_classes, num_channels):
    '''
    layers = [
        nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    ]
    '''
    LSTMmodel = LSTM_emorec()
    '''
    if isinstance(num_classes, list):
        encoders = [_Encoder(layers=layers) for _ in num_classes]
        return [_Model(output_size=cls, encoder=encoder) for cls, encoder in zip(num_classes, encoders)]
    else:
        encoder = _Encoder(layers=layers)
        return _Model(output_size=num_classes, encoder=encoder)
    '''
    if isinstance(num_classes, list):
        encoders = [_Encoder( model=LSTMmodel ) for _ in num_classes]
        return [_Model(output_size=cls, encoder=encoder) for cls, encoder in zip(num_classes, encoders)]
    else:
        encoder = _Encoder( model=LSTMmodel )
        return _Model(output_size=num_classes, encoder=encoder)
