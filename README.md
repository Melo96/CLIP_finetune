# CLIP_finetune

The project objective is to fine-tune CLIP (Contrastive Language-Image Pre-Training) on web-crawled image and its relevant search query. 

## Install
```bash
conda create --name clip-ft python=3.7
pip install -r requirements.txt
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```