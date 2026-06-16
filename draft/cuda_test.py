import torch
print("CUDA 是否可用：", torch.cuda.is_available())
print("当前设备：", torch.cuda.current_device())
print("GPU 名称：", torch.cuda.get_device_name())
import torch

x = torch.rand(3, 3).to("cuda")
print(x.device)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)
