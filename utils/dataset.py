import numpy as np
import pandas as pd
import os
from io import BytesIO
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset, BatchSampler
import random
import pdb

# Reference: https://www.kaggle.com/code/zacchaeus/clip-finetune
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class MyDataSet(Dataset):
    """LayoutLM dataset with visual features."""
    def __init__(self, df, processor, query2label, max_length=32):
        self.df = df
        self.processor = processor
        self.query2label = query2label
        self.max_seq_length = max_length
        self.return_tensors = 'pt'
 
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx,:].to_dict()
        file_id = item["id"]
        query = query = "a picture relevant to " + item["query"]
        img_path = item["img_path"]
        
        with open(img_path, 'rb') as f:
            img = pickle.load(f)['image']

        image = Image.open(BytesIO(img)).convert("RGB")

        encoding = self.processor(text=query, images=image, return_tensors=self.return_tensors, 
                                max_length=self.max_seq_length, padding='max_length', truncation=True)

        encoding['input_ids'] = encoding['input_ids'][0]
        encoding['attention_mask'] = encoding['attention_mask'][0]
        encoding['pixel_values'] = encoding['pixel_values'][0]
        encoding['label'] = self.query2label[query]

        return encoding