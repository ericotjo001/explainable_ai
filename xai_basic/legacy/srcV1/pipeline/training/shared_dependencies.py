import os, sys, time, random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

VERBOSE_THRESHOLD = 100
this_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 