import torch
import torch.nn as nn


class BiLSTMWordSegmentation(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Linear(
            hidden_dim * 2,
            2
        )

    def forward(self, input_ids):
        # input_ids: [batch_size, sequence_length]
        embedded = self.embedding(input_ids)

        # [batch_size, sequence_length, hidden_dim * 2]
        encoded, _ = self.lstm(embedded)

        # We predict boundaries between characters.
        # Therefore, for sequence length N,
        # we need predictions for N - 1 gaps.
        gap_features = encoded[:, :-1, :] + encoded[:, 1:, :]

        # [batch_size, sequence_length - 1, 2]
        logits = self.classifier(gap_features)

        return logits


if __name__ == "__main__":
    model = BiLSTMWordSegmentation(
        vocab_size=159
    )

    dummy_input = torch.randint(
        low=0,
        high=159,
        size=(4, 124)
    )

    output = model(dummy_input)

    print("BiLSTM model created successfully!")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Expected output shape: [4, 123, 2]")