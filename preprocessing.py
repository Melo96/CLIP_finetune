from pyarrow import feather
from pathlib import Path
from io import BytesIO
from tqdm import tqdm, trange
import pandas as pd
import pickle
from PIL import Image
from utils.preprocessing_image import preprocess_image



def main():
    """
    Steps of preprocessing:
        1. Load "train.feather".
        2. To fine-tune CLIP, we need to create image-text pairs, where the text captions is something related to the image (our "query"). So, only data with "is_relevant=1" is kept in training and validation. 
        3. Download image from image source URLs. Save them as bytearray in a pickle file. 
        4. Save processed data as ".feather".

    Args:
        data_path (Path): path to data to be processed.
    """
    data_path = Path('/data/kaiwen/ahrefs/dataset')
    relevant_columns = ['id', 'query', 'url_page', 'src', 'title', 'alt', 'is_relevant'] # columns to keep for fine-tuning
    df = feather.read_feather(data_path / 'train.feather').filter(items=relevant_columns)

    relevant_df = df.loc[df['is_relevant']==1].reset_index(drop=True)

    failed = []
    df_save = []
    # some images are blank, which will cause error
    corrupted = ['af408364133664b68836a9a474717ce5', 'dbccc72ea88ef231a123ce9c08745244']

    save_path = data_path / "train_data"
    save_path.mkdir(exist_ok=True, parents=True)

    preprocess_image(relevant_df, save_path)

    for i in trange(len(relevant_df)):
        file_id = relevant_df['id'][i]
        query = relevant_df['query'][i]
        folder_num = i // 1000
        save_path_final = save_path / f'{(folder_num+1)*1000}/{file_id}.pkl'
        if save_path_final.is_file() and file_id not in corrupted:
            with open(save_path_final, 'rb') as f:
                img = pickle.load(f)['image']
            try:
                # make sure the image can be opened
                image = Image.open(BytesIO(img)).convert("RGB")
                row = {'id': file_id, 'query': query, 'img_path': str(save_path_final)}
                df_save.append(row)
            except:
                failed.append(file_id)
                continue
    
    df_save = pd.DataFrame(df_save).reset_index(drop=True)
    print(len(df_save))
    df_save.to_feather(data_path / "train_final_df.feather")
        
if __name__=="__main__":
    main()