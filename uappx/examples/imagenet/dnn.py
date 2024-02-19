# import os, time, joblib
# import numpy as np
import torch
import torch.nn as nn
# from src.utils import parse_bool_from_string, strbool_description, readjust_bools
import torchvision.models as mod
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
normalizeImageTransform = transforms.Compose([transforms.ToTensor(), 
    normalizeTransform])