def build_lstm(input_size=1, hidden_size=64, num_layers=2, output_size=1):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install PyTorch with pip install -e '.[torch]'") from exc
    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.1 if num_layers > 1 else 0)
            self.head = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            sequence, _ = self.lstm(x)
            return self.head(sequence[:, -1, :])
    return LSTMForecaster()
