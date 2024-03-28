from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from tools import make_label
import numpy as np
import ase
import csv
import pandas as pd





def input_reshape():

    with open('VT.pkl', 'rb') as file_handle:
        data = pickle.load(file_handle)
    with open('tricky_attributes.pkl', 'rb') as file_handle:
        tricky = pickle.load(file_handle)
    with open('element_attributes.pkl', 'rb') as file_handle:
        element_properties = pickle.load(file_handle)
    # ############################### input reshape #########################################
    dir =  './cif/'
    i=1
    AD={}
    All_FT_data = make_label(dir, i, data, tricky, element_properties, 'latest2')
    ALL_data=All_FT_data
    K=list(ALL_data['FP'])
    ta=pd.read_csv('./id_prop.csv')
    for i in range(len(K)):
        records = K[i]
        result = []  
        for y in range(0, 11):  
          for x in range(0, 6): 
            if x == 0:
             result.append([])
            result[y].append(records[x + y * 6]) 
        AD[ta['name'][i]]=result
    with open('TEST' + '.pkl', "wb") as f:
        pickle.dump(AD, f)