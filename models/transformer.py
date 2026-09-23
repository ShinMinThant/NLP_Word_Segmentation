
import torch
import torch.nn as nn


class TransformerWordSegmentation(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_length=256
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Learnable positional embeddings
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Combine the representations of adjacent characters
        # to predict the boundary between them.
        self.classifier = nn.Linear(
            embedding_dim * 2,
            2
        )

        self.dropout = nn.Dropout(dropout)

        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):

        # input_ids:
        # [batch_size, sequence_length]

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum length {self.max_length}."
            )

        # Character embeddings
        x = self.embedding(input_ids)

        # Position IDs
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device
        ).unsqueeze(0).expand(
            batch_size,
            sequence_length
        )

        # Add positional information
        x = x + self.position_embedding(position_ids)

        x = self.dropout(x)

        # Transformer expects True for positions that should
        # be ignored, so invert the attention mask.
        padding_mask = None

        if attention_mask is not None:
            padding_mask = attention_mask == 0

        # Transformer Encoder
        encoded = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        # Represent each gap using the two adjacent characters.
        left = encoded[:, :-1, :]
        right = encoded[:, 1:, :]

        gap_features = torch.cat(
            [left, right],
            dim=-1
        )

        # Predict:
        # 0 = No boundary
        # 1 = Word boundary
        logits = self.classifier(gap_features)

        return logits


if __name__ == "__main__":

    model = TransformerWordSegmentation(
        vocab_size=159
    )

    dummy_input = torch.randint(
        low=0,
        high=159,
        size=(4, 124)
    )

    dummy_mask = torch.ones(
        4,
        124,
        dtype=torch.long
    )

    output = model(
        dummy_input,
        dummy_mask
    )

    print("Transformer model created successfully!")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Expected output shape: [4, 123, 2]")
