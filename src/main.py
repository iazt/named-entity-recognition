from torchtext import data, datasets
import torch
from torch import nn
from src.rnn import RNNNer
from src.utils import train_loop
import torch.optim as optim

directory = '/home/ignacio/'

# Primer Field: TEXT. Representan los tokens de la secuencia
TEXT = data.Field(lower=False)

# Segundo Field: NER_TAGS. Representan los Tags asociados a cada palabra.
NER_TAGS = data.Field(unk_token=None, is_target=True)

fields = (("text", TEXT), ("nertags", NER_TAGS))


train_data = datasets.SequenceTaggingDataset(directory + 'train_NER_esp.txt', fields, encoding="iso-8859-1", separator=" ")
valid_data = datasets.SequenceTaggingDataset(directory + 'val_NER_esp.txt', fields, encoding="iso-8859-1", separator=" ")
test_data = datasets.SequenceTaggingDataset(directory + 'test_NER_esp.txt', fields, encoding="iso-8859-1", separator=" ")

TEXT.build_vocab(train_data)
NER_TAGS.build_vocab(train_data)

BATCH_SIZES = (32,2,2)

# Usar cuda si es que está disponible.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dividir datos entre entrenamiento y test
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_sizes = BATCH_SIZES,
    device = device,
    sort=False)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
OUTPUT_DIM = len(NER_TAGS.vocab)
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.3
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

# Creamos nuestro modelo.
model = RNNNer(INPUT_DIM,
                     EMBEDDING_DIM,
                     HIDDEN_DIM,
                     OUTPUT_DIM,
                     N_LAYERS,
                     BIDIRECTIONAL,
                     DROPOUT,
                     PAD_IDX)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


model.apply(init_weights)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# Optimizador
optimizer = optim.Adam(model.parameters())

TAG_PAD_IDX = NER_TAGS.vocab.stoi[NER_TAGS.pad_token]

# Loss: Cross Entropy
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

# Enviamos el modelo y la loss a cuda (en el caso en que esté disponible)
model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 1
train_loop(N_EPOCHS, model, train_iterator, valid_iterator, optimizer, criterion)

