import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from pyarrow import feather
from pathlib import Path
from tqdm import tqdm, trange
from io import BytesIO
from PIL import Image
import pickle

data_path = Path('/data/kw/data/ahrefs')

divide_factor = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
pretrained_name = 'openai/clip-vit-base-patch32'
model = CLIPModel.from_pretrained(pretrained_name) # load pretrained model
# model.to(device)
processor = CLIPProcessor.from_pretrained(pretrained_name) # load model processor that help us prepare the data input

test_data = feather.read_feather(data_path / 'test_final_df.feather')
submission = pd.read_csv(data_path / 'submission.csv')

predictions = {}

for i in trange(len(test_data)):
    file_id = test_data['id'][i]
    query = "a picture relevant to " + test_data['query'][i]
    folder_num = i // divide_factor
    save_path_final = data_path / f'test_data/{(folder_num+1)*divide_factor}/{file_id}.pkl'
    if save_path_final.is_file():
        with open(save_path_final, 'rb') as f:
            img = pickle.load(f)['image']
        
        image = Image.open(BytesIO(img)).convert("RGB").resize((224, 224))
        encoding = processor(text=query, images=image, return_tensors="pt", padding=True)
        
        outputs = model(**encoding)
        text_embed, image_embed = outputs.text_embeds[0], outputs.image_embeds[0]
        dot_product = torch.dot(text_embed, image_embed).item()
        
        predictions[file_id] = dot_product

for i in range(len(submission)):
    file_id = submission['id'][i]
    if file_id in predictions:
        submission['is_relevant'][i] = predictions[file_id]

submission.to_csv(data_path / 'submission_done.csv')