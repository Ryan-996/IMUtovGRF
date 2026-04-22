import torch
import torch.nn as nn


class CNN_BiLSTM(nn.Module):
    def __init__(
        self,
        input_channels=78,
        conv1_out_channels=256,
        conv2_out_channels=128,
        lstm_hidden_size=256,
        lstm_layers=3,
        dropout_rate=0.1,
        output_dim=2
    ):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv1_out_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        # BiLSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=conv2_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        # Time-step-wise regression head
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 60)
        self.bn_fc1 = nn.BatchNorm1d(60)

        self.fc2 = nn.Linear(60, 30)
        self.bn_fc2 = nn.BatchNorm1d(30)

        self.fc3 = nn.Linear(30, output_dim)

    def forward(self, x):
        """
        x: shape (N, T, C)
        return: shape (N, T, output_dim)
        """

        # Conv1d expects (N, C, T)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.dropout(x)

        # Back to (N, T, C) for LSTM
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.dropout(x)

        # Fully connected layers applied at each time step
        x = self.fc1(x)
        x = self.bn_fc1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn_fc2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(x)

        x = self.fc3(x)

        return x