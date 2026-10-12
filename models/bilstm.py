import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids, lengths=None):
        # input_ids shape: [batch_size, sequence_length]
        embedded = self.embedding(input_ids)

        if lengths is not None:
            # Ignore padded characters inside the BiLSTM.
            packed = pack_padded_sequence(
                embedded,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False
            )

            packed_encoded, _ = self.lstm(packed)

            encoded, _ = pad_packed_sequence(
                packed_encoded,
                batch_first=True,
                total_length=input_ids.size(1)
            )
        else:
            # Kept for compatibility with older code.
            encoded, _ = self.lstm(embedded)

        # Predict boundaries between adjacent characters.
        gap_features = encoded[:, :-1, :] + encoded[:, 1:, :]
        logits = self.classifier(gap_features)

        return logits


if __name__ == "__main__":
    model = BiLSTMWordSegmentation(vocab_size=159)

    dummy_input = torch.randint(1, 159, (4, 124))
    dummy_lengths = torch.tensor([124, 100, 80, 60])

    output = model(dummy_input, dummy_lengths)

    print("BiLSTM model created successfully!")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Expected output shape: [4, 123, 2]")
