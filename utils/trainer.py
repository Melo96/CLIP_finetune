from tqdm import tqdm
from transformers import TrainingArguments, Trainer
import torch
from torch import nn
from torchmetrics import AUROC

class train_and_evaluate:
    def __init__(self, device):
        self.device = device
    
    def comput_loss(self, logits, labels):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1), labels.view(-1))
        return loss
        
    def train(self, train_dataloader, model, optimizer):
        steps = len(train_dataloader)
        total_loss = 0
        tk0 = tqdm(train_dataloader, total=steps)
        model.train()
        for bi, batch in enumerate(tk0):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)
            outputs = model(pixel_values = pixel_values,input_ids=input_ids, attention_mask=attention_mask, return_loss=True)
            loss = outputs.loss
            total_loss+=loss.item()
            loss.backward()
            optimizer.step()
        return outputs, total_loss / steps
    
    def eval(self, val_dataloader, model):
        steps = len(val_dataloader)
        tk0 = tqdm(val_dataloader, total=steps)
        total_loss = 0
        model.eval()
        for bi, batch in enumerate(tk0):
            with torch.no_grad():
                input_ids=batch['input_ids'].to(self.device)
                attention_mask=batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                outputs = model(pixel_values = pixel_values,input_ids=input_ids, attention_mask=attention_mask, return_loss=True)
                total_loss+=outputs.loss.item()
        return total_loss / steps
