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
from sklearn.metrics import mean_absolute_error
from pre_process import pre_process
from input_reshape import input_reshape
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp=pre_process()
ir=input_reshape()
feature_filename = "./TEST.pkl"
feature_dicts = joblib.load(open(feature_filename, "rb"))

all_data = pd.read_csv("id_prop.csv")

names= all_data.name.values
existing_names = [name for name in names if str(name) in feature_dicts]

# Create a new DataFrame that only includes the existing names
all_data = all_data[all_data.name.isin(existing_names)]
names= all_data.name.values
dts=[]

for name in names:
    for lst in feature_dicts[str(name)]:
        for i in range(len(lst)):
            if lst[i] is None:
                lst[i] = 0.0
    dt = torch.Tensor(feature_dicts[str(name)])
    dts.append(dt)
data_tensor = torch.stack(dts)

labels = all_data.energy.values
label_tensor = torch.Tensor(labels).reshape(-1)
# Partition the data set
train_data, test_data, train_labels, test_labels, train_names, test_names = train_test_split(
    data_tensor, label_tensor, names, test_size=0.1, random_state=42)

dataset = data.TensorDataset(train_data, train_labels)
for i, (datat, label) in enumerate(dataset):
    if torch.isnan(datat).any():
        print(f"Found NaN in data at index {i}, replacing with 0.0")
        datat = torch.nan_to_num(datat, nan=0.0)
        dataset.tensors = (torch.cat([dataset.tensors[0][:i], datat[None, :], dataset.tensors[0][i+1:]]), dataset.tensors[1])

train_loader = data.DataLoader(dataset, batch_size=700, shuffle=True)
test_dataset = data.TensorDataset(test_data, test_labels)
for i, (datat, label) in enumerate(test_dataset):
    if torch.isnan(datat).any():
        print(f"Found NaN in data at index {i}, replacing with 0.0")
        datat = torch.nan_to_num(datat, nan=0.0)
        dataset.tensors = (torch.cat([dataset.tensors[0][:i], datat[None, :], dataset.tensors[0][i+1:]]), dataset.tensors[1])
test_loader = data.DataLoader(test_dataset, batch_size=700, shuffle=False)

model=ResNet()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

b,c=train_model(model, criterion, optimizer, train_loader, num_epochs=600)

model.eval()
all_preds = []
all_labels = []
test_names_list = [] 

with torch.no_grad():
    for i, (batch_data, batch_labels) in enumerate(test_loader):
        preds = model(batch_data)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch_labels.cpu().numpy())
        batch_names = test_names[i*test_loader.batch_size:(i+1)*test_loader.batch_size]
        test_names_list.extend(batch_names)

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
data_dict = {'name': test_names_list, 'true': all_labels, 'pre': all_preds}     
df = pd.DataFrame(data_dict)
df.to_csv('./results.csv', index=False)

# MAE
test_mae = mean_absolute_error(all_labels, all_preds)
print(f"Test MAE: {test_mae}")

torch.save(model.state_dict(), 'model.pth')


