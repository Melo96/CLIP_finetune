import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from pyarrow import feather
from pathlib import Path
from tqdm import tqdm, trange
from io import BytesIO
import PIL
from PIL import Image
import pickle

data_path = Path('/data/kw/data/ahrefs')

divide_factor = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
pretrained_name = 'openai/clip-vit-base-patch32'
model = CLIPModel.from_pretrained(pretrained_name) # load pretrained model
model.to(device)
processor = CLIPProcessor.from_pretrained(pretrained_name) # load model processor that help us prepare the data input

test_data = feather.read_feather(data_path / 'test.feather')
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
        
        try:
            image = Image.open(BytesIO(img)).convert("RGB").resize((224, 224))
        # skip corrupted images
        except PIL.UnidentifiedImageError:
            continue
        except OSError:
            print("Image truncated ", file_id, folder_num)
            continue
        
        encoding = processor(text=query, images=image, return_tensors="pt", padding=True)
        
        input_ids=encoding['input_ids'].to(device)
        attention_mask=encoding['attention_mask'].to(device)
        pixel_values = encoding['pixel_values'].to(device)
        outputs = model(pixel_values = pixel_values,input_ids=input_ids, attention_mask=attention_mask)
        text_embed, image_embed = outputs.text_embeds[0], outputs.image_embeds[0]
        dot_product = torch.dot(text_embed, image_embed).item()
    
        submission.loc[i, 'is_relevant'] = dot_product

print("max similarity: ", max(submission['is_relevant'].values))
print("min similarity: ", min(submission['is_relevant'].values))

submission.to_csv(data_path / 'submission_done.csv')