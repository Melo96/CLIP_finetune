import requests
from io import BytesIO
from cairosvg import svg2png
from pathlib import Path
from tqdm import trange, tqdm
import json
import multiprocessing
import os
import collections
import pickle
import re


def preprocess_image(df, save_path, overwrite=False, divide_factor=1000, max_workers=4):
    """
    Steps for image preprocessing:
        - Create a multiprocessing session (maximum 4 workers) to accelerate the process.
        - Define user agents. 
        - For every image URL, remove all text after image file extension to get the original image.
        - Request the image and save it as bytearray in a pickle file. 

    Args:
        df (DataFrame): data to be processed. 
        save_path (Path): path to save the processed data.
        overwrite (bool, optional): whether to overwrite existing files. Defaults to False.
        divide_factor (int, optional): create and save data to new folder for every {divide_factor} of images.
                                        Defaults to 1000.
    """
    workers = min(max_workers, os.cpu_count() // 2)
    print(f"Number of workers: {workers}")

    overwrite = False # whether to overwrite existing data
    divide_factor = 1000 # number to split for new folder when saving images data. 

    # count error
    url_load_failed = []
    img_load_failed = []

    # use multiprocessing to accelerate processing
    with multiprocessing.Pool(processes=workers) as pool:
        jobs = []
        status_code_count = collections.defaultdict(lambda: [])
        for row_id in trange(len(df)):
            folder_num = row_id // divide_factor
            save_path_final = save_path / f'{(folder_num+1)*divide_factor}'
            save_path_final.mkdir(exist_ok=True, parents=True)
            file_id = df['id'][row_id]
            query = df['query'][row_id]
            img_url = df['src'][row_id]
            
            if not overwrite and (save_path_final / f'{file_id}.pkl').is_file():
                continue
            
            job = pool.apply_async(load_and_save, args=(file_id, query, img_url, save_path_final))
            jobs.append(job)

        for job in tqdm(jobs):
            error_type, file_id = job.get()
            if error_type=='url request error':
                url_load_failed.append(file_id)
            elif error_type=='img loading error':
                img_load_failed.append(file_id)
            elif error_type!='success':
                status_code_count[error_type].append(file_id)

    print("Number of URL requesting errors:", len(url_load_failed))
    print("Number of image loading error:", len(img_load_failed))
    print("Number of website response errors:", len(dict(status_code_count).values()))
    total_error = len(url_load_failed) + len(img_load_failed) + len(dict(status_code_count).values())
    print("Total number of error:", total_error)
    print("Total number of success:", len(df)-total_error)

    with open(save_path / 'url_load_failed.json', 'w') as f:
        json.dump(url_load_failed, f)

    with open(save_path / 'img_load_failed.json', 'w') as f:
        json.dump(img_load_failed, f)
        
    with open(save_path / 'status_code_count.json', 'w') as f:
        json.dump(dict(status_code_count), f)


def load_and_save(file_id, query, img_url, save_path_final, 
                  user_agent='Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
                  ):
    # filter variables in the URL to get the original image
    parsed_url = re.search(r'(.*\.(jpg|JPG|png|PNG|svg|npj|webp|gif))(.*)', img_url)
    img_url = parsed_url.group(1) if parsed_url else img_url
    surfix = img_url.split(".")[-1]

    try:
        response = requests.get(img_url, headers={"User-Agent": user_agent}, timeout=10, stream=True)
    except:
        return 'url request error', file_id
        
    if response.status_code==200:
        # convert ".svg" to ".png"
        if surfix=='svg':
            try:
                img = svg2png(bytestring=BytesIO(response.content).read())
            except:
                return 'img loading error', file_id
        else:
            try:
                img = BytesIO(response.content).getvalue()
            except:
                return 'img loading error', file_id
    else:
        return response.status_code, file_id

    to_save = {'id': file_id, 'query': query, 'image': img}
    with open(save_path_final / f'{file_id}.pkl', 'wb') as f:
        pickle.dump(to_save, f)
    return 'success', file_id

