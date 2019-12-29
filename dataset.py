from torch.utils.data import Dataset
from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import torch
import nltk
import re


class SimpleDataset(Dataset):
    ID_COL_NAME = "Id"
    PREDICT_COL_NAME = "Task 2"
    TRAIN_COL = [col.lower() for col in [ID_COL_NAME, "Abstract", PREDICT_COL_NAME]]
    CLASS = ("THEORETICAL", "ENGINEERING", "EMPIRICAL",)

    def __init__(self, data_path, pretrained, train=False, reduce_data_rate=1, worker=cpu_count() - 2):
        assert 0 < reduce_data_rate <= 1

        super().__init__()
        self.train = train
        self.pretrained = pretrained

        # Worker Preprocess
        self.dataset = list()
        dataset = pd.read_csv(data_path)
        size = dataset.shape[0]
        if reduce_data_rate < 1:
            dataset = pd.read_csv(data_path).iloc[:int(size * reduce_data_rate)]
            size = dataset.shape[0]
        split_dataset = list()
        start_idx = 0
        for end_idx in range(size // worker + 1, size, size // worker + 1):
            split_dataset.append(dataset.iloc[start_idx:end_idx])
            start_idx = end_idx
        if start_idx != size:
            split_dataset.append(dataset.iloc[start_idx:])
        with Pool(processes=worker) as pool:
            result_list = list()
            for i in range(worker):
                result_list.append(pool.apply_async(self.preprocess, (split_dataset[i], self.pretrained, self.train,)))
            for i in range(worker):
                self.dataset += result_list[i].get()

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def restore_word(cls, words):
        words = nltk.pos_tag(words)
        wnl = nltk.stem.WordNetLemmatizer()
        new_words = list()
        for word, tag in words:
            if tag.startswith('NN'):
                new_words.append(wnl.lemmatize(word, pos='n'))
            elif tag.startswith('VB'):
                new_words.append(wnl.lemmatize(word, pos='v'))
            elif tag.startswith('JJ'):
                new_words.append(wnl.lemmatize(word, pos='a'))
            elif tag.startswith('R'):
                new_words.append(wnl.lemmatize(word, pos='r'))
            else:
                new_words.append(word)
        return new_words

    @classmethod
    def preprocess(cls, dataset, pretrained, train=False):
        new_dataset = pd.DataFrame()

        # Remove useless columns
        for col in dataset.columns:
            if col.lower() in cls.TRAIN_COL:
                new_dataset[col] = dataset[col]

        # Lower
        for col in new_dataset.columns:
            if col not in [cls.ID_COL_NAME, cls.PREDICT_COL_NAME]:
                new_dataset[col] = new_dataset[col].str.lower()
                new_dataset[col] = new_dataset[col].str.replace("\'", "")
                new_dataset[col] = new_dataset[col].str.replace("-", " ")
                new_dataset[col] = new_dataset[col].str.replace(",", "")
                new_dataset[col] = new_dataset[col].str.replace(".", "")

        # Tokenizer
        datas = list()
        tokenizer = BertTokenizer.from_pretrained(pretrained)
        for row in tqdm(new_dataset.iterrows(), total=new_dataset.shape[0], desc=f"Tokenizing", ncols=70):
            row = row[1]
            data = dict()
            if train:
                if "OTHERS" in row[cls.PREDICT_COL_NAME].upper():
                    continue
                data[cls.PREDICT_COL_NAME] = torch.tensor([1 if _class in row[cls.PREDICT_COL_NAME].upper() else 0 for _class in cls.CLASS], dtype=torch.float)
            data[cls.ID_COL_NAME] = row[cls.ID_COL_NAME]
            data["Abstract"] = [" ".join(cls.restore_word([word for word in sentence.split() if re.match(r"[a-z]+", word)])) for sentence in row["Abstract"].split("$$$")[:6]]
            data["Abstract"] = cls.bert_tokenizer(data["Abstract"], tokenizer)

            datas.append(data)

        return datas

    @classmethod
    def bert_tokenizer(cls, sentences, tokenizer):
        first_len = None
        word_pieces = ["[CLS]"]
        for sentence in sentences:
            word_pieces += tokenizer.tokenize(sentence) + ["[SEP]"]
            if not first_len:
                first_len = len(word_pieces)
        tokens = torch.tensor((tokenizer.convert_tokens_to_ids(word_pieces)))
        segments = torch.tensor([0] * first_len + [1] * (len(word_pieces) - first_len), dtype=torch.long)

        return tokens, segments

    @classmethod
    def unique_predict_class(cls, dataset):
        class_set = set()
        for each in dataset[cls.PREDICT_COL_NAME]:
            for _cls in each.split():
                class_set.add(_cls)
        return class_set


if __name__ == "__main__":
    dataset = SimpleDataset("task2_trainset.csv", worker=7, train=True)
    dataset.dataset[len(dataset) - 1]
