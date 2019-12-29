from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from dataset import SimpleDataset
from model import SequenceClassification
import torch
import numpy as np
import os

BASE_DIR = "save"


def train_mini_batch(samples):
    ids = list()
    abstract_token = list()
    abstract_seg = list()
    labels = list()
    for sample in samples:
        ids.append(sample["Id"])
        abstract_token.append(sample["Abstract"][0])
        abstract_seg.append(sample["Abstract"][1])
        if "Task 2" in sample.keys():
            labels.append(sample["Task 2"])

    tokens_tensors = pad_sequence(abstract_token, batch_first=True)
    segments_tokens = pad_sequence(abstract_seg, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    if not len(labels):
        return tokens_tensors, segments_tokens, masks_tensors
    else:
        return tokens_tensors, segments_tokens, masks_tensors, torch.stack(labels, 0)


def f1_score(logits, labels):
    label_num = len(labels[0])
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(logits)):
        for j in range(label_num):
            if logits[i][j] == labels[i][j] == 1:
                TP += 1
            elif logits[i][j] == 1 and labels[i][j] == 0:
                FP += 1
            elif logits[i][j] == 0 and labels[i][j] == 1:
                FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2 * precision * recall / (precision + recall)


def evaluate(model, dataloader, device, threshold=0.4, value=False):
    outputs_ = list()
    labels_ = list()
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluate", ncols=70):
            tokens_tensors, segments_tokens, masks_tensors, labels = [t.to(device) for t in data]
            outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tokens, attention_mask=masks_tensors)
            logits = torch.sigmoid(outputs)
            logits = (logits > threshold) * 1.0
            outputs_ += logits.cpu().numpy().astype(np.int).tolist()
            labels_ += labels.cpu().numpy().astype(np.int).tolist()
    return f1_score(outputs_, labels_)


def run_epoch(model, dataloader, optimizer, criterion):
    running_loss = 0.0
    model.train()
    for _, data in tqdm(enumerate(dataloader), total=len(dataloader), ncols=70):
        tokens_tensors, segments_tokens, masks_tensors, labels = [t.to(device) for t in data]
        optimizer.zero_grad()
        outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tokens, attention_mask=masks_tensors)
        loss = criterion(outputs, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()
    f1_train = evaluate(model, trainloader, device)
    f1_test = evaluate(model, testloader, device)

    print(f"""
Loss_Avg: {running_loss / len(trainloader)}
Train F1: {f1_train}
Test F1: {f1_test}""")
    return f1_train, f1_test


if __name__ == '__main__':
    BATCH_SIZE = 8
    EPOCHS = 20
    NUM_LABELS = 3
    DROPOUT = 0.1
    PRETRAINED = "bert-base-uncased"
    MODEL_NAME = "Bert(large)_lr5e-6(.5-5)"

    dataset = SimpleDataset("task2_trainset.csv", PRETRAINED, train=True, reduce_data_rate=0.1)

    test_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size
    traindata, testdata = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_mini_batch)
    testloader = DataLoader(testdata, batch_size=BATCH_SIZE, collate_fn=train_mini_batch)
    model = SequenceClassification.from_pretrained("bert-large-uncased", num_labels=NUM_LABELS)
    model.config.hidden_dropout_prob = DROPOUT

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        print(f"\n[ Epoch {ep + 1}/{EPOCHS} ]")
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-6 * (0.5 ** (ep // 5)))
        train_acc, test_acc = run_epoch(model, trainloader, optimizer, criterion)

        if ep > 1:
            if not os.path.exists(os.path.join(BASE_DIR, MODEL_NAME)):
                os.mkdir(os.path.join(BASE_DIR, MODEL_NAME))
            if not os.path.exists(os.path.join(BASE_DIR, MODEL_NAME, f"ep{ep + 1}")):
                os.mkdir(os.path.join(BASE_DIR, MODEL_NAME, f"ep{ep + 1}"))
            model.save_pretrained(os.path.join(BASE_DIR, MODEL_NAME, f"ep{ep + 1}"))
            with open(os.path.join(BASE_DIR, MODEL_NAME, f"ep{ep + 1}", "acc.txt"), "w") as f:
                f.write(f"Train_F1: {train_acc}\nTest_F1: {test_acc}")
            print(f"\n{'#' * 5} Model saved {'#' * 5}")
    print("Done.")
