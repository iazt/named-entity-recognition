# Definimos las métricas
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def calculate_metrics(preds, y):
    """
    Calcula precision, recall y f1 de cada batch.
    """

    # Obtener el indice de la clase con probabilidad mayor. (clases)
    y_pred = preds.argmax(dim = 1, keepdim = True)
    # Obtenemos los indices distintos de 0.

    y_pred = y_pred.view(-1).to('cpu')
    y_true = y.to('cpu')

    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true,y_pred)
    return precision, recall, f1, accuracy


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_acc = 0
    i = 0
    model.train()

    # Por cada batch del iterador de la época:
    for batch in iterator:
        # Extraemos el texto y los tags del batch que estamos procesado
        text = batch.text
        tags = batch.nertags

        # Reiniciamos los gradientes calculados en la iteración anterior
        optimizer.zero_grad()

        # text = [sent len, batch size]

        # Predecimos los tags del texto del batch.
        predictions = model(text)

        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]

        # Reordenamos los datos para calcular la loss

        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]

        # Calculamos el Cross Entropy de las predicciones con respecto a las etiquetas reales
        loss = criterion(predictions, tags)

        # Calculamos el accuracy
        precision, recall, f1, accuracy = calculate_metrics(predictions, tags)

        # Calculamos los gradientes
        loss.backward()

        # Actualizamos los parámetros de la red
        optimizer.step()

        # Actualizamos el loss y las métricas
        epoch_loss += loss.item()
        epoch_precision += precision
        epoch_recall += recall
        epoch_f1 += f1
        epoch_acc += accuracy

    return epoch_loss / len(iterator), epoch_precision / len(iterator), epoch_recall / len(iterator), epoch_f1 / len(
        iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_acc = 0

    model.eval()

    # Indicamos que ahora no guardaremos los gradientes
    with torch.no_grad():
        # Por cada batch
        for batch in iterator:
            text = batch.text
            tags = batch.nertags

            # Predecimos
            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            # Calculamos el Cross Entropy de las predicciones con respecto a las etiquetas reales
            loss = criterion(predictions, tags)

            # Calculamos las métricas
            precision, recall, f1, accuracy = calculate_metrics(predictions, tags)

            # Actualizamos el loss y las métricas
            epoch_loss += loss.item()
            epoch_precision += precision
            epoch_recall += recall
            epoch_f1 += f1
            epoch_acc += accuracy

    return epoch_loss / len(iterator), epoch_precision / len(iterator), epoch_recall / len(iterator), epoch_f1 / len(
        iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_loop(n_epochs, model, train_iterator, valid_iterator, optimizer, criterion):
    best_valid_f1 = float(0)

    for epoch in range(n_epochs):

        start_time = time.time()

        # Recuerdo: train_iterator y valid_iterator contienen el dataset dividido en batches.

        # Entrenar
        train_loss, train_precision, train_recall, train_f1, train_acc = train(model, train_iterator, optimizer,
                                                                               criterion)

        # Evaluar
        valid_loss, valid_precision, valid_recall, valid_f1, val_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Si obtuvimos mejores resultados, guardamos este modelo en el almacenamiento (para poder cargarlo luego)
        # Si detienen el entrenamiento prematuramente, pueden cargar el modelo en el siguiente recuadro de código.
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), 'modelo_tarea_2.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train f1: {train_f1:.2f} | Train precision: {train_precision:.2f} | Train recall: {train_recall:.2f} | Train_accuracy: {train_acc:.2f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. f1: {valid_f1:.2f} | Val. precision: {valid_precision:.2f} | Val. recall: {valid_recall:.2f} | Val. accuracy: {val_acc:.2f}')