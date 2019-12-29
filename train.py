from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from dataset import SimpleDataset
from model import SequenceClassification
import torch


def create_mini_batch(samples):
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


def predictions(model, dataloader, device, threshold=0.4, value=False):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluate", ncols=70):
            tokens_tensors, segments_tokens, masks_tensors, labels = [t.to(device) for t in data]
            outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tokens, attention_mask=masks_tensors)
            logits = torch.sigmoid(outputs)
            logits = (logits > threshold) * 1.0
            for i in range(logits.shape[0]):
                total += 1
                if torch.all(torch.eq(logits[i], labels[i])):
                    correct += 1
    return correct, total


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
    train_correct, train_total = predictions(model, trainloader, device)
    test_correct, test_total = predictions(model, testloader, device)

    print(f"""
Loss_Avg: {running_loss / len(trainloader)}
Train_Acc: {int(train_correct / train_total * 100)}%, Correct: {train_correct}, Total: {train_total}
Test_Acc: {int(test_correct / test_total * 100)}%, Correct: {test_correct}, Total: {test_total}
""")
    return int(train_correct / train_total * 100), int(test_correct / test_total * 100)


if __name__ == '__main__':
    BATCH_SIZE = 8
    EPOCHS = 30
    NUM_LABELS = 4

    # reduce for testing
    # dataset = SimpleDataset("task2_trainset.csv", train=True, reduce_data_rate=0.1)

    dataset = SimpleDataset("task2_trainset.csv", train=True)

    test_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size
    traindata, testdata = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True, collate_fn=create_mini_batch)
    testloader = DataLoader(testdata, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    model = SequenceClassification.from_pretrained("bert-large-uncased", num_labels=NUM_LABELS)

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        print(f"\n[ Epoch {ep + 1}/{EPOCHS} ]")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5 * (0.1 ** (ep // 15)))
        train_acc, test_acc = run_epoch(model, trainloader, optimizer, criterion)

        if (ep + 1) % 5 == 0:
            torch.save(model.state_dict(), f"save\\ep{ep + 1}[train_acc{train_acc}test_acc{test_acc}].pt")