import torch, os, subprocess, platform

print("torch.cuda.is_available :", torch.cuda.is_available())
print("GPU name               :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print(subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True).decode() if platform.system()=="Linux" else "")

print("Torch version          :", torch.__version__)
print("CUDA version           :", torch.version.cuda)
print("cuDNN version          :", torch.backends.cudnn.version())
print("Torch module path      :", torch.__file__)
