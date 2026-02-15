from torch import nn, zeros
from torch.nn.modules.dropout import Dropout


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_dim,
        drop=0.1,
        batch_first=True,
    ):
        super(LSTM, self).__init__()
        # self.norm = nn.BatchNorm1d(24)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first)
        self.drop = Dropout(drop)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, output_dim)
        self.act2 = nn.Tanh()

    def forward(self, x):
        # self.norm(x)
        output, _ = self.lstm(x)
        output = self.drop(output)
        output = self.linear(output)
        output = self.act(output)
        output = self.linear2(output)
        output = self.act2(output)

        return output

    # maybe add one more layer a
