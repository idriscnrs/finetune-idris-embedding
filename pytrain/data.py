from functools import partial
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class IdrisPairsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        """Return the number of element of the dataset"""
        return len(self.df)

    def __getitem__(self, idx) -> tuple[str, str]:
        """Return the input for the model and the label for the loss"""
        df_elem = self.df.iloc[idx]
        question = df_elem["question"]
        context = df_elem["context"]
        return question, context


class FaqsPairsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        """Return the number of element of the dataset"""
        return len(self.df)

    def __getitem__(self, idx) -> tuple[str, str, str]:
        """Return the input for the model and the label for the loss"""
        df_elem = self.df.iloc[idx]
        question = df_elem["question"]
        answer_text = df_elem["answer_text"]
        answer_md = df_elem["answer_md"]
        return question, answer_text, answer_md


def collate_fn(
    batch: list[tuple[str, str]], tokenizer: PreTrainedTokenizer
) -> BatchEncoding:
    batch = [sentence for pairs in batch for sentence in pairs]
    model_inp = tokenizer(
        batch, padding=True, truncation=True, return_tensors='pt', max_length=128
    )
    return model_inp


def filter_df(
    df: pd.DataFrame,
    keep_faqs: bool,
    tresh_ground: int,
    tresh_rel: int,
    tresh_stand: int,
) -> pd.DataFrame:
    # Remove doc from faq
    if not keep_faqs:
        df = df.drop(df[df["source"].str.contains("/faqs/")].index)

    df = df.drop(df[df["groundedness_prompt_rating"] <= tresh_ground].index)
    df = df.drop(df[df["relevant_prompt_rating"] <= tresh_rel].index)
    df = df.drop(df[df["standalone_prompt_rating"] <= tresh_stand].index)
    return df


def get_dataloader(
    data_dir: Path,
    train_csv_name: Path,
    valid_csv_name: Path,
    batch_size: int,
    keep_faqs: bool,
    tresh_ground: int,
    tresh_rel: int,
    tresh_stand: int,
    tokenizer: PreTrainedTokenizer
) -> tuple[DataLoader, DataLoader]:
    train_dataframe = pd.read_csv(data_dir / train_csv_name)
    valid_dataframe = pd.read_csv(data_dir / valid_csv_name)

    train_dataframe = filter_df(
        train_dataframe, keep_faqs, tresh_ground, tresh_rel, tresh_stand
    )

    train_dataset = IdrisPairsDataset(train_dataframe)
    valid_dataset = FaqsPairsDataset(valid_dataframe)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        prefetch_factor=1,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=1,
        prefetch_factor=1,
        drop_last=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    return train_dataloader, valid_dataloader
