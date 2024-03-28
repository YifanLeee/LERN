import pickle
import pandas as pd
from tqdm import tqdm
from ase.io import read
import numpy as np
import os
from tools import fingerprint_adslab, _concatenate_shell,fingerprint_element_attributes

def pre_process():
    # combine full data
    dir = './cif/'
    data = []
    result = pd.read_csv('./id_prop.csv')
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    file=os.listdir(dir) 

    for index in tqdm(file):
        index=str(index).strip('.cif')
        atoms = read(dir + str(index)+'.cif')
        case_energy = ((result.loc[result['name']==index])['energy']).values[0]
        case_VT = fingerprint_adslab(atoms)
        case_VT2 = _concatenate_shell(case_VT)
        case = {'index': index,
                'atoms': atoms,
                'energy': case_energy,
                'VT': case_VT,
                'VT2': case_VT2
                }
        data.append(case)

    for i in tqdm(range(len(data))):
        try:
            case_VT = fingerprint_adslab(data[i]['atoms'])
            data[i]['VT'] = case_VT
        except:
            print('wrong ' + str(i))
    with open('VT' + '.pkl', "wb") as f:
        pickle.dump(data, f)

    for case in tqdm(data):
        shells = _concatenate_shell(case['VT'])
        case['VT2'] = shells

    f_num = []
    s_num = []
    t_num = []
    for case in data:
        f_num.append(case['VT2']['n_1_layer'])
        s_num.append(case['VT2']['n_2_layer'])
        t_num.append(case['VT2']['n_3_layer'])
    error_index = []
    for i in range(len(f_num)):
        if f_num[i] == 0:
            print(i)
            error_index.append(i)

    for index in tqdm(error_index):
        VT = fingerprint_adslab(data[index]['atoms'])
        shells = _concatenate_shell(VT)
        if shells['n_1_layer'] == 0:
            print(index)

    check = []
    for index in error_index:
        check.append(data[index]['VT'])

    # combine all sets
    total_doc = []

    ############################ inspect number of cases by layer category###########################
    import matplotlib.pyplot as plt

    n_first = []
    n_second = []
    n_third = []
    for case in data:
        n_first.append(case['VT2']['n_1_layer'])
        n_second.append(case['VT2']['n_2_layer'])
        n_third.append(case['VT2']['n_3_layer'])
    n_first = set(n_first)
    n_second = set(n_second)
    n_third = set(n_third)

    data_layer = {
        '111': [], '121': [], '131': [], '141': [],
        '211': [], '221': [], '231': [], '241': [],
        '311': [], '321': [], '331': [], '341': [],
        '112': [], '122': [], '132': [], '142': [],
        '212': [], '222': [], '232': [], '242': [],
        '312': [], '322': [], '332': [], '342': [],
        '113': [], '123': [], '133': [], '143': [],
        '213': [], '223': [], '233': [], '243': [],
        '313': [], '323': [], '333': [], '343': [],
        '114': [], '124': [], '134': [], '144': [],
        '214': [], '224': [], '234': [], '244': [],
        '314': [], '324': [], '334': [], '344': [],
    }

    for case in data:
        layers_index = str(case['VT2']['n_1_layer']) + str(case['VT2']['n_2_layer'])+ str(case['VT2']['n_3_layer'])
        data_layer[layers_index].append(case)
    x = np.array(list(data_layer.keys()))
    y = []
    for item in data_layer:
        y.append(len(data_layer[item]))

    ################ calculate several average energy attributes (tricky)  of element #################
    # case_num is the number of case that include certain element
    # atom num is total number of certain element in all cases
    # total energy is sum energy of all cases that include certain element
    # everage_atom_energy is ev/atom in a case
    # total eae is total

    # get name array
    element_array = {'first_layer': [], 'second_layer': [], 'third_layer': [], 'all_layers': []}
    for item in data:
        for key1 in item['VT2']['first_layer']:
            element_array['first_layer'].append(key1)
            element_array['all_layers'].append(key1)
        for key2 in item['VT2']['second_layer']:
            element_array['second_layer'].append(key2)
            element_array['all_layers'].append(key2)
        for key3 in item['VT2']['third_layer']:
            element_array['third_layer'].append(key3)
            element_array['all_layers'].append(key3)
    element_array = {'first_layer': set(element_array['first_layer']),
                    'second_layer': set(element_array['second_layer']),
                    'third_layer': set(element_array['third_layer']),
                    'all_layers': set(element_array['all_layers'])}
    # first layer
    first_layer_conclusion = {}
    for element in element_array['first_layer']:
        case_num = 0
        atom_num = 0
        total_energy = 0
        total_eae = 0
        all_energis = []
        for item in data:
            for atom in item['VT2']['first_layer']:
                if str(element) == atom:
                    case_num = case_num + 1
                    atom_num = atom_num + item['VT2']['first_layer'][atom]['number']
                    total_energy = total_energy + item['energy']
                    all_energis.append(item['energy'])
                    everage_atom_energy = (item['energy']) / (item['VT2']['first_layer'][atom]['number'])
                    total_eae = total_eae + everage_atom_energy
        conclusion = {'case_num': case_num, 'atom_num': atom_num, 'total_energy': total_energy,
                    'total_eae': total_eae, 'case_median': np.median(all_energis), 'aver_case': total_energy / case_num,
                    'aver_atom': total_energy / atom_num, 'aver_eae': total_eae / case_num}
        first_layer_conclusion[str(element)] = conclusion
    # second layer
    second_layer_conclusion = {}
    for element in element_array['second_layer']:
        case_num = 0
        atom_num = 0
        total_energy = 0
        total_eae = 0
        all_energis = []
        for item in data:
            for atom in item['VT2']['second_layer']:
                if str(element) == atom:
                    case_num = case_num + 1
                    atom_num = atom_num + item['VT2']['second_layer'][atom]['number']
                    total_energy = total_energy + item['energy']
                    all_energis.append(item['energy'])
                    everage_atom_energy = (item['energy']) / (item['VT2']['second_layer'][atom]['number'])
                    total_eae = total_eae + everage_atom_energy
        conclusion = {'case_num': case_num, 'atom_num': atom_num, 'total_energy': total_energy,
                    'total_eae': total_eae, 'case_median': np.median(all_energis), 'aver_case': total_energy / case_num,
                    'aver_atom': total_energy / atom_num, 'aver_eae': total_eae / case_num}
        second_layer_conclusion[str(element)] = conclusion
    # third layer
    third_layer_conclusion = {}
    for element in element_array['third_layer']:
        case_num = 0
        atom_num = 0
        total_energy = 0
        total_eae = 0
        all_energis = []
        for item in data:
            for atom in item['VT2']['third_layer']:
                if str(element) == atom:
                    case_num = case_num + 1
                    atom_num = atom_num + item['VT2']['third_layer'][atom]['number']
                    total_energy = total_energy + item['energy']
                    all_energis.append(item['energy'])
                    everage_atom_energy = (item['energy']) / (item['VT2']['third_layer'][atom]['number'])
                    total_eae = total_eae + everage_atom_energy
        conclusion = {'case_num': case_num, 'atom_num': atom_num, 'total_energy': total_energy,
                    'total_eae': total_eae, 'case_median': np.median(all_energis), 'aver_case': total_energy / case_num,
                    'aver_atom': total_energy / atom_num, 'aver_eae': total_eae / case_num}
        third_layer_conclusion[str(element)] = conclusion

    all_layers_conclusion = {}
    for element in element_array['all_layers']:
        case_num = 0
        atom_num = 0
        total_energy = 0
        total_eae = 0
        all_energis = []
        for item in data:
            if (element in item['VT2']['first_layer']) or (element in item['VT2']['second_layer']) or (element in item['VT2']['third_layer']):
                case_num = case_num + 1
                total_energy = total_energy + item['energy']
                all_energis.append(item['energy'])
                if (element in item['VT2']['first_layer']) and (element in item['VT2']['second_layer']) and (element in item['VT2']['third_layer']):
                    atom_num = atom_num + item['VT2']['first_layer'][element]['number'] + \
                            item['VT2']['second_layer'][element]['number'] + \
                            item['VT2']['third_layer'][element]['number']
                    everage_atom_energy = (item['energy']) / (
                                item['VT2']['second_layer'][element]['number'] + item['VT2']['first_layer'][element]['number'] + item['VT2']['third_layer'][element]['number'])
                elif element in item['VT2']['first_layer']:
                    atom_num = atom_num + item['VT2']['first_layer'][element]['number']
                    everage_atom_energy = (item['energy']) / (item['VT2']['first_layer'][element]['number'])
                elif element in item['VT2']['second_layer']:
                    atom_num = atom_num + item['VT2']['second_layer'][element]['number']
                    everage_atom_energy = (item['energy']) / (item['VT2']['second_layer'][element]['number'])
                else:
                    atom_num = atom_num + item['VT2']['third_layer'][element]['number']
                    everage_atom_energy = (item['energy']) / (item['VT2']['third_layer'][element]['number'])
        conclusion = {'case_num': case_num, 'atom_num': atom_num, 'total_energy': total_energy,
                    'total_eae': total_eae, 'case_median': np.median(all_energis), 'aver_case': total_energy / case_num,
                    'aver_atom': total_energy / atom_num, 'aver_eae': total_eae / case_num}
        all_layers_conclusion[str(element)] = conclusion

    trick_attributes = {'first_layer': first_layer_conclusion,
                        'second_layer': second_layer_conclusion,
                        'third_layer': third_layer_conclusion,
                        'all_layers': all_layers_conclusion}
    with open("tricky_attributes.pkl", "wb") as f:
        pickle.dump(trick_attributes, f)

    # ######################## get other properties of each element################
    element_properties = {}
    for element in element_array['all_layers']:
        element_properties[element] = fingerprint_element_attributes(element)

    for item in element_properties:
        if element_properties[item]['elemental properties']['electron_affinity'] is None:
            element_properties[item]['elemental properties']['electron_affinity'] = 0
    for item in element_properties:
        del element_properties[item]['macro properties']['mineral_hardness']

    with open("element_attributes.pkl", "wb") as f:
        pickle.dump(element_properties, f)