# IdemiaProject

This project is carried out in the context of the Artificial Intelligence Masters of **TelecomParis**.

<sub>Made with __Python__</sub>

## Project

Development of an AI evaluating the position of a possibly occulted pedestrian from a picture.

## How to setup?

- First, clone the repository:

```
git clone https://github.com/PierreBio/IdemiaProject.git
```

- Then go to the root of the project:

```
cd IdemiaProject
```

- Create a virtual environment:

```
py -m venv venv
```

- Activate your environment:

```
.\venv\Scripts\activate
```

- Install requirements:

```
pip install -r requirements.txt
```

## How to launch?

- Once the project is setup, you can launch DataParser script to parse data:

```
py -m bin.DataParser
```

- You can launch RunModel script to train a model after parsing data:

```
py -m bin.RunModel
```

## How to use your GPU
Download CUDA toolkit : https://developer.nvidia.com/cuda-downloads
You can check your version and if the installation was successfull with:
```
nvcc --version
```

Install pytorch with CUDA (remove it if it's already installed without):
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Now you can check if it will use your GPU with this python code:
```
import torch
print(torch.cuda.is_available())
```

## How to download COCO Dataset?
Go to https://cocodataset.org/#download
Download 2017 Train/Val annotations

## How to download OCHuman database
Go to https://github.com/liruilong940607/OCHumanApi?tab=readme-ov-file
Download link is at the bottom of page
Fill in your information and click on Send, you will have access to Google Drive
Download files and save to a OCHuman folder
Use bin/OCParser.py to create the csv files

## Ressources
