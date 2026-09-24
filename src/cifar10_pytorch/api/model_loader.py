import torch

from cifar10_pytorch.config import MODEL_PATH
from cifar10_pytorch.model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()