import torch.nn as nn
import torch.nn.functional as F


# Definir la red
class RNNNer(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        super().__init__()

        # Capa de embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx, )

        # Capa LSTM
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=0.3)

        # Capa de salida
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, text):
        # text = [sent len, batch size]

        # Convertir lo enviado a embedding
        embedded = F.relu(self.dropout(self.embedding(text)))

        # embedded = [sent len, batch size, emb dim]

        # Pasar los embeddings por la rnn (LSTM)
        outputs, (hidden, cell) = self.rnn(embedded)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # Predecir usando la capa de salida.
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]

        return predictions
