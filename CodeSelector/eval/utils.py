import os
import numpy as np
import pandas as pd
import pickle
import time
import datetime 
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
from transformers import BertTokenizer
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import  AdamW, BertConfig
warnings.filterwarnings('ignore')

def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second 
    elapsed_rounded = int(round(elapsed))
    # format as hh:mm:ss 
    return str(datetime.timedelta(seconds=elapsed_rounded))


