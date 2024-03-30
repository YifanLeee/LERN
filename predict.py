import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from copy import deepcopy, copy
import torch.utils.data as data
from model import *
import random
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pre_process import pre_process
from input_reshape import input_reshape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_adsorption_energy():

    pp=pre_process()

    ir=input_reshape()

    feature_filename = "./TEST.pkl"
    feature_dicts = joblib.load(open(feature_filename, "rb"))

    all_data = pd.read_csv("./id_prop.csv")

    names= all_data.name.values

    data_tensor = torch.stack([torch.Tensor(feature_dicts[str(name)]) for name in names])

    labels = all_data.energy.values
    label_tensor = torch.Tensor(labels).reshape(-1)
    dataset = data.TensorDataset(data_tensor, label_tensor)
    pre_loader = data.DataLoader(dataset, batch_size=700)

    # create model instance
    model = ResNet()

    # load model parameters
    model.load_state_dict(torch.load('model.pth'))

    # Set the model to evaluation mode
    model.eval()

    # input data for prediction
    for input, label in pre_loader:
        output = model(input)

    pre=list(np.array(output.detach().numpy()))
    tre=list(np.array(label.detach().numpy()))
    # save output regression result
    pre=pd.DataFrame({'name':names,'pre':pre})
    pre.to_csv('predict.csv',encoding='gbk')
