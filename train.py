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


def f1_score(logits, labels, ep=1e-8):
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
    precision = TP / (TP + FP + ep)
    recall = TP / (TP + FN + ep)
    return 2 * precision * recall / (precision + recall + ep)


def evaluate(model, dataloader, device, threshold=[0.4, 0.4, 0.4], myrange=None):
    outputs_ = list()
    labels_ = list()
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluate", ncols=70):
            tokens_tensors, segments_tokens, masks_tensors, labels = [t.to(device) for t in data]
            outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tokens, attention_mask=masks_tensors)
            logits = torch.sigmoid(outputs)
            outputs_ += logits.cpu().numpy().tolist()
            labels_ += labels.cpu().numpy().astype(np.int).tolist()
        outputs_ = np.array(outputs_)
        if myrange is not None:
            best_threshold = [None, None, None]
            score = 0
            for t1 in tqdm(np.arange(myrange[0], myrange[1], myrange[2]), ncols=70, desc=f"Threshold"):
                for t2 in np.arange(myrange[0], myrange[1], myrange[2]):
                    for t3 in np.arange(myrange[0], myrange[1], myrange[2]):
                        t1, t2, t3 = np.round(t1, 2), np.round(t2, 2), np.round(t3, 2)
                        bool_matrix = np.zeros(outputs_.shape)
                        bool_matrix[:, 0] = (outputs_[:, 0] > t1).astype(np.int)
                        bool_matrix[:, 1] = (outputs_[:, 1] > t2).astype(np.int)
                        bool_matrix[:, 2] = (outputs_[:, 2] > t3).astype(np.int)
                        t_outputs_ = bool_matrix.tolist()
                        t_score = f1_score(t_outputs_, labels_)
                        if t_score > score:
                            best_threshold = [t1, t2, t3]
                            score = t_score
            return score, best_threshold
        else:
            outputs_[:, 0] = (outputs_[:, 0] > threshold[0]).astype(np.int)
            outputs_[:, 1] = (outputs_[:, 1] > threshold[1]).astype(np.int)
            outputs_[:, 2] = (outputs_[:, 2] > threshold[2]).astype(np.int)
            return f1_score(outputs_.tolist(), labels_)


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
    f1_test, best_TH = evaluate(model, testloader, device, myrange=(0.4, 0.61, 0.01,))
    f1_train = evaluate(model, trainloader, device, threshold=best_TH)

    print(f"""
Loss_Avg: {running_loss / len(trainloader)}
Train F1: {f1_train}
Test F1: {f1_test}
Best Threshold: {best_TH}""")
    return f1_train, f1_test, best_TH


if __name__ == '__main__':
    BATCH_SIZE = 8
    EPOCHS = 8
    NUM_LABELS = 3
    DROPOUT = 0.3
    PRETRAINED = "bert-base-uncased"
    MODEL_NAME = "Bert(base_large)_lr1e-6(.5-5)d310TH"

    dataset = SimpleDataset("task2_trainset.csv", PRETRAINED, train=True)

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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6 * (0.5 ** (ep // 5)))
        train_acc, test_acc, best_TH = run_epoch(model, trainloader, optimizer, criterion)

        if ep > 1:
            if not os.path.exists(os.path.join(BASE_DIR, MODEL_NAME)):
                os.mkdir(os.path.join(BASE_DIR, MODEL_NAME))
            if not os.path.exists(os.path.join(BASE_DIR, MODEL_NAME, f"ep{ep + 1}")):
                os.mkdir(os.path.join(BASE_DIR, MODEL_NAME, f"ep{ep + 1}"))
            model.save_pretrained(os.path.join(BASE_DIR, MODEL_NAME, f"ep{ep + 1}"))
            with open(os.path.join(BASE_DIR, MODEL_NAME, f"ep{ep + 1}", "acc.txt"), "w") as f:
                f.write(f"Train_F1: {train_acc}\nTest_F1: {test_acc}\nBest Threshold: {best_TH}")
            print(f"\n{'#' * 5} Model saved {'#' * 5}")
    print("Done.")
