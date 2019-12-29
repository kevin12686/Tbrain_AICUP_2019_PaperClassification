from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from dataset import SimpleDataset
from model import SequenceClassification
import numpy as np
import torch
import pandas as pd


def create_mini_batch(samples):
    ids = list()
    abstract_token = list()
    abstract_seg = list()
    for sample in samples:
        ids.append(sample["Id"])
        abstract_token.append(sample["Abstract"][0])
        abstract_seg.append(sample["Abstract"][1])

    tokens_tensors = pad_sequence(abstract_token, batch_first=True)
    segments_tokens = pad_sequence(abstract_seg, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tokens, masks_tensors, ids


def predictions(model, dataloader, device, threshold=0.4):
    counter = 0
    zero = torch.zeros(4).to(device)
    id_list = list()
    flag_list = list()
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluate", ncols=70):
            tokens_tensors, segments_tokens, masks_tensors = [t.to(device) for t in data[:-1]]
            ids = data[-1]
            outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tokens, attention_mask=masks_tensors)
            logits = torch.sigmoid(outputs)
            logits = (logits > threshold) * 1.0
            id_list += ids
            flag_list += logits.cpu().numpy().astype(np.int).tolist()
            for i in range(logits.shape[0]):
                if torch.all(torch.eq(logits[i], zero)):
                    counter += 1
    print(f"zero count: {counter}")
    return id_list, flag_list


def prepare_submit(ids, flags, class_order):
    submit_df = pd.DataFrame(columns=["order_id", "THEORETICAL", "ENGINEERING", "EMPIRICAL", "OTHERS"])
    for i in tqdm(range(len(ids)), total=len(ids), desc="Output"):
        if flags[3] == 0:
            if flags[i][0] == flags[i][1] == flags[i][2] == 0:
                flags[i][3] = 1
        # submit_df.loc[i] = [ids[i], flags[2], flags[0], flags[1], 1 if flags[0] == flags[1] == flags[2] == 0 else 0]
        submit_df.loc[i] = [ids[i], flags[i][2], flags[i][0], flags[i][1], flags[i][3]]
    submit_df.to_csv("summitForm.csv")


if __name__ == '__main__':
    NUM_LABELS = 4
    BATCH_SIZE = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SimpleDataset("task2_public_testset.csv", reduce_data_rate=1)
    loader = DataLoader(dataset, BATCH_SIZE, collate_fn=create_mini_batch)
    model = SequenceClassification.from_pretrained("bert-large-uncased", num_labels=NUM_LABELS)
    model.load_state_dict(torch.load("save\\ep25[train_acc99test_acc40].pt"))
    model.to(device)
    ids, flags = predictions(model, loader, device)
    prepare_submit(ids, flags, dataset.CLASS)
