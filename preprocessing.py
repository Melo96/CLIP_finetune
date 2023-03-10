from pyarrow import feather
from pathlib import Path
from io import BytesIO
from tqdm import tqdm, trange
import pandas as pd
import pickle
from PIL import Image
import argparse
import json
from utils.preprocessing_image import preprocess_image


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        required=True,
                        type=str,
                        help='Path to data to be processed.'
                        )
    parser.add_argument("--file_name",
                        required=False,
                        type=str,
                        help='Name of the data (train or test). Default train',
                        default='train'
                        )
    parser.add_argument("--overwrite",
                        required=False,
                        type=bool,
                        help='Whether to overwrite existing data.',
                        default=False
                        )
    parser.add_argument("--divide_factor",
                        required=False,
                        type=int,
                        help='Create and save data to new folder for every {divide_factor} of images. Defaults to 1000',
                        default=1000
                        )
    parser.add_argument("--max_workers",
                        required=False,
                        type=int,
                        help='Maximum number of workers in multi-processing. Defaults to 4',
                        default=4
                        )
    args = parser.parse_args()
    return args

def main(data_path, file_name='train', overwrite=False, divide_factor=1000, max_workers=4):
    """
    Steps of preprocessing:
        1. Load "train.feather".
        2. To fine-tune CLIP, we need to create image-text pairs, where the text captions is something related to the image (our "query"). So, only data with "is_relevant=1" is kept in training and validation. 
        3. Download image from image source URLs. Save them as bytearray in a pickle file. 
        4. Save processed data as ".feather".

    Args:
        data_path (Path): path to data to be processed.
    """
    relevant_columns = ['id', 'query', 'url_page', 'src', 'title', 'alt', 'is_relevant'] # columns to keep for fine-tuning
    df = feather.read_feather(data_path / f'{file_name}.feather').filter(items=relevant_columns)

    if file_name=="train":
        relevant_df = df.loc[df['is_relevant']==1].reset_index(drop=True)
    else:
        relevant_df = df

    failed = []
    df_save = []
    # some images are blank, which will cause error
    corrupted = ['af408364133664b68836a9a474717ce5', 'dbccc72ea88ef231a123ce9c08745244']

    save_path = data_path / f"{file_name}_data"
    save_path.mkdir(exist_ok=True, parents=True)

    preprocess_image(relevant_df, save_path, overwrite, divide_factor, max_workers)

    for i in trange(len(relevant_df)):
        file_id = relevant_df['id'][i]
        query = relevant_df['query'][i]
        folder_num = i // divide_factor
        save_path_final = save_path / f'{(folder_num+1)*divide_factor}/{file_id}.pkl'
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
    print(len(failed))
    df_save.to_feather(data_path / f"{file_name}_final_df.feather")
    with open(data_path / f"{file_name}_failed.json", "w") as f:
        json.dump(failed, f)

if __name__=="__main__":
    args = get_args_parser()
    data_path = args.data_path
    file_name = args.file_name
    overwrite = args.overwrite
    divide_factor = args.divide_factor
    max_workers = args.max_workers
    # data_path = Path('/Users/yangkaiwen/Documents/data/ahrefs interview/dataset')
    # file_name = 'test'
    # overwrite = False
    # divide_factor = 1000
    # max_workers = 4
    main(data_path, file_name, overwrite, divide_factor, max_workers)