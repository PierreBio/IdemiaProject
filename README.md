# IdemiaProject
Development of an AI evaluating distance between camera and a pedestrian.

# How to use GPU
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
