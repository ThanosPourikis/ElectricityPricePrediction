from torch.nn.modules.dropout import Dropout
from torch.nn import BatchNorm1d, LSTM, Linear, Tanh, Module, Conv1d


class Hybrid_LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_dim,
        drop=0.1,
        batch_first=True,
    ):
        super(Hybrid_LSTM, self).__init__()
        self.conv1 = Conv1d(24, 24, 2)
        self.lstm = LSTM(1, hidden_size, num_layers, batch_first)
        self.drop = Dropout(drop)
        self.linear = Linear(hidden_size, hidden_size)
        self.act = Tanh()
        self.linear2 = Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        output, _ = self.lstm(x)
        output = self.drop(output)
        output = self.linear(output)
        output = self.act(output)
        output = self.linear2(output)

        return output

    # maybe add one more layer a
