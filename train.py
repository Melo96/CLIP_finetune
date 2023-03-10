import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from pyarrow import feather
from pathlib import Path
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm, trange
import math
import argparse
from sklearn.model_selection import train_test_split

from utils.dataset import MyDataSet, BalancedBatchSampler
from utils.trainer import train_and_evaluate

import pdb

random_state = 10

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        required=True,
                        type=str,
                        help='Path to data to be processed.'
                        )
    parser.add_argument("--model_save_name",
                        required=False,
                        type=str,
                        default="clip_test",
                        help='Name of saved model and results.'
                        )
    parser.add_argument("--pretrained_name",
                        required=False,
                        type=str,
                        default='openai/clip-vit-base-patch32',
                        help='Path to pretrained model.'
                        )
    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        default=16,
                        help='Batch size. Default 16. '
                        )
    args = parser.parse_args()
    return args

def finetune(model, train_dataloader, val_dataloader, trainer, model_save_name):
    MODEL_PATH = model_save_name + ".bin"
    BEST_PATH = model_save_name + "_best.bin"
    HISTORY_PATH = model_save_name + ".csv"
    
    history = []
    lr = 1e-5
    epoch_num = 10
    # weight_decay = 5e-2
    optimizer = optim.AdamW(model.parameters(), lr=lr, 
                        #   weight_decay=weight_decay
                            )
    lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=5)

    min_loss = math.inf
    for epoch in trange(epoch_num):
        outputs, train_loss = trainer.train(train_dataloader, model, optimizer, lr_scheduler)
        val_loss = trainer.eval(val_dataloader, model)
        # torch.save(model.state_dict(), MODEL_PATH)
        if val_loss <= min_loss:
            torch.save(model.state_dict(), BEST_PATH)
            min_loss = val_loss

        history.append([epoch+1, train_loss, val_loss])
        print("epoch:", epoch+1,
                "train_loss:", train_loss,
                "validation_loss:", val_loss,
                "lr:", optimizer.param_groups[0]['lr']
                )
    history_df = pd.DataFrame(history, columns=["epoch", "train_loss", "validation_loss"])
    history_df.to_csv(HISTORY_PATH) 

    torch.save(model.state_dict(), MODEL_PATH)
    
def get_train_test_split(df, query2label):
    index = [i for i in range(len(df))]
    labels = [query2label["a picture relevant to " + df['query'][i]] for i in range(len(df))]
    
    train_index, val_index, train_labels, val_labels = train_test_split(index, labels, test_size=0.1, random_state=random_state, shuffle=True)
    
    train_df = []
    val_df = []
    for i in range(len(df)):
        row = df.iloc[i]
        if i in train_index:
            train_df.append(row)
        elif i in val_index:
            val_df.append(row)

    train = pd.DataFrame(train_df).reset_index(drop=True)
    print(f"train data size: {len(train)}")
    valid = pd.DataFrame(val_df).reset_index(drop=True)
    print(f"val data size: {len(valid)}")
    return train, valid, train_labels, val_labels

def main(data_path, model_save_name, pretrained_name='openai/clip-vit-base-patch32', batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(random_state)
    BATCH_SIZE = batch_size

    total_df = feather.read_feather(data_path / 'train_final_df.feather') # read preprocessed data
    query2label = {"a picture relevant to " + q: l for l, q in enumerate(total_df['query'].unique())} # create map for query to label
    train, valid, train_labels, val_labels = get_train_test_split(total_df, query2label) # split train and validation set

    model = CLIPModel.from_pretrained(pretrained_name) # load pretrained model
    model.to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_name) # load model processor that help us prepare the data input

    train_dataset = MyDataSet(train, processor, query2label)
    train_sampler = BalancedBatchSampler(torch.tensor(train_labels), BATCH_SIZE, 1) # create balanced batch
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    
    val_dataset = MyDataSet(valid, processor, query2label)
    val_sampler = BalancedBatchSampler(torch.tensor(val_labels), BATCH_SIZE, 1)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)

    trainer = train_and_evaluate(device=device)
    
    finetune(model, train_dataloader, val_dataloader, trainer, model_save_name)
            
if __name__=="__main__":
    args = get_args_parser()
    data_path = args.data_path
    model_save_name = args.model_save_name
    pretrained_name = args.pretrained_name
    batch_size = args.batch_size
    # data_path = Path('/data/kw/data/ahrefs')
    # model_save_name = "clip_test"
    # pretrained_name = 'openai/clip-vit-base-patch32'
    # batch_size = 16
    main(data_path, model_save_name, pretrained_name, batch_size)