from ase.constraints import FixAtoms
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from collections import Counter
from mendeleev import element
from pymatgen.core import periodic_table as pmpt
import math
import operator
from tqdm import tqdm
import pandas as pd

def calculate_distance(H_position, distanceSet):
    result_distance = []
    for item in distanceSet:
        dis = math.sqrt((H_position['x'] - item['x']) ** 2 + (H_position['y'] - item['y']) ** 2 \
                        +(H_position['z'] - item['z']) ** 2 )
        result_distance.append(dis)
    return result_distance

def calculate_curvature(H_position, distanceSet):
    diff_x = 0
    diff_y = 0
    diff_z = 0
    for item in distanceSet:
        diff_x = diff_x + item['x'] - H_position['x']
        diff_y = diff_y + item['y'] - H_position['y']
        diff_z = diff_z + item['z'] - H_position['z']
    curvature = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
    return curvature

def constrain_slab(atoms, z_cutoff=3.):
    '''
    This function fixes sub-surface atoms of a slab. Also works on systems that
    have slabs + adsorbate(s), as long as the slab atoms are tagged with `0`
    and the adsorbate atoms are tagged with positive integers.

    Inputs:
        atoms       ASE-atoms class of the slab system. The tags of these atoms
                    must be set such that any slab atom is tagged with `0`, and
                    any adsorbate atom is tagged with a positive integer.
        z_cutoff    The threshold to see if slab atoms are in the same plane as
                    the highest atom in the slab
    Returns:
        atoms   A deep copy of the `atoms` argument, but where the appropriate
                atoms are constrained
    '''
    # Work on a copy so that we don't modify the original
    atoms = atoms.copy()

    # We'll be making a `mask` list to feed to the `FixAtoms` class. This list
    # should contain a `True` if we want an atom to be constrained, and `False`
    # otherwise
    mask = []

    # If we assume that the third component of the unit cell lattice is
    # orthogonal to the slab surface, then atoms with higher values in the
    # third coordinate of their scaled positions are higher in the slab. We make
    # this assumption here, which means that we will be working with scaled
    # positions instead of Cartesian ones.
    scaled_positions = atoms.get_scaled_positions()
    unit_cell_height = np.linalg.norm(atoms.cell[2])

    # If the slab is pointing upwards, then fix atoms that are below the
    # threshold
    if atoms.cell[2, 2] > 0:
        max_height = max(position[2] for position, atom in zip(scaled_positions, atoms)
                         if atom.tag == 0)
        threshold = max_height - z_cutoff / unit_cell_height
        for position, atom in zip(scaled_positions, atoms):
            if atom.tag == 0 and position[2] < threshold:
                mask.append(True)
            else:
                mask.append(False)

    # If the slab is pointing downwards, then fix atoms that are above the
    # threshold
    elif atoms.cell[2, 2] < 0:
        min_height = min(position[2] for position, atom in zip(scaled_positions, atoms)
                         if atom.tag == 0)
        threshold = min_height + z_cutoff / unit_cell_height
        for position, atom in zip(scaled_positions, atoms):
            if atom.tag == 0 and position[2] > threshold:
                mask.append(True)
            else:
                mask.append(False)

    else:
        raise RuntimeError('Tried to constrain a slab that points in neither '
                           'the positive nor negative z directions, so we do '
                           'not know which side to fix')

    atoms.constraints += [FixAtoms(mask=mask)]
    return atoms


def remove_adsorbate(adslab):
    '''
    This function removes adsorbates from an adslab and gives you the locations
    of the binding atoms. Note that we assume that the first atom in each adsorbate
    is the binding atom.

    Arg:
        adslab  The `ase.Atoms` object of the adslab. The adsorbate atom(s) must
                be tagged with non-zero integers, while the slab atoms must be
                tagged with zeroes. We assume that for each adsorbate, the first
                atom (i.e., the atom with the lowest index) is the binding atom.
    Returns:
        slab                The `ase.Atoms` object of the bare slab.
        binding_positions   A dictionary whose keys are the tags of the
                            adsorbates and whose values are the cartesian
                            coordinates of the binding site.
    '''
    # Operate on a local copy so we don't propagate changes to the original
    slab = adslab.copy()

    # Remove all the constraints and then re-constrain the slab. We do this
    # because ase does not like it when we delete atoms with constraints.
    slab.set_constraint()
    slab = constrain_slab(slab)

    # Delete atoms in reverse order to preserve correct indexing
    binding_positions = {}
    for i, atom in reversed(list(enumerate(slab))):
        if atom.tag != 0:
            binding_positions[atom.tag] = atom.position
            del slab[i]

    return slab, binding_positions


def __get_coordination_string(nn_info):
    '''
    This helper function takes the output of the `VoronoiNN.get_nn_info` method
    and gives you a standardized coordination string.

    Arg:
        nn_info     The output of the
                    `pymatgen.analysis.local_env.VoronoiNN.get_nn_info` method.
    Returns:
        coordination    A string indicating the coordination of the site
                        you fed implicitly through the argument, e.g., 'Cu-Cu-Cu'
    '''
    coordinated_atoms = [neighbor_info['site'].species_string
                         for neighbor_info in nn_info
                         if neighbor_info['site'].species_string != 'H']

    cdn_atoms_position = []
    for item in nn_info:
        if item['site'].species_string != 'H':
            cdn_atoms_position.append({'element':item['site'].species_string,
                           'atom_index':item['site'].index,
                           'x':item['site'].x,
                           'y':item['site'].y,
                           'z':item['site'].z})

    coordination = '-'.join(sorted(coordinated_atoms))
    return coordination, cdn_atoms_position

def fingerprint_adslab(atoms):
    '''
    This function will fingerprint a slab+adsorbate atoms object for you.
    Currently, it only works with one adsorbate.

    Arg:
        atoms   `ase.Atoms` object to fingerprint. The slab atoms must be
                tagged with 0 and adsorbate atoms must be tagged with
                non-zero integers.  This function also assumes that the
                first atom in each adsorbate is the binding atom (e.g.,
                of all atoms with tag==1, the first atom is the binding;
                the same goes for tag==2 and tag==3 etc.).
    Returns:
        fingerprint A dictionary whose keys are:
                        coordination            A string indicating the
                                                first shell of
                                                coordinated atoms
                        neighborcoord           A list of strings
                                                indicating the coordination
                                                of each of the atoms in
                                                the first shell of
                                                coordinated atoms
                        nextnearestcoordination A string identifying the
                                                coordination of the
                                                adsorbate when using a
                                                loose tolerance for
                                                identifying "neighbors"
    '''
    # Replace the adsorbate[s] with a single Uranium atom at the first binding
    # site. We need the Uranium there so that pymatgen can find its
    # coordination.

    struct = AseAtomsAdaptor.get_structure(atoms)
    atomic_number = struct.atomic_numbers
    H_index = 0
    for i in range(len(atomic_number)):
        if atomic_number[i] == 1:
            H_index = i

    H_position = {'x': struct.sites[H_index].x,
                  'y': struct.sites[H_index].y,
                  'z': struct.sites[H_index].z}
    #
    # We have a standard and a loose Voronoi neighbor finder for various
    # purposes
    vnn = VoronoiNN(allow_pathological=False,tol=0.5, cutoff=15)
    vnn_loose = VoronoiNN(allow_pathological=False,tol=0.2, cutoff=15)

    # Find the coordination
    nn_info = vnn.get_nn_info(struct, n=H_index)
    coordination, coord_position = __get_coordination_string(nn_info)

    # Find the neighborcoord
    neighbor_all_nn_info =[]
    for neighbor_info in nn_info:
        neighbor_index = neighbor_info['site_index']
        neighbor_nn_info = vnn_loose.get_nn_info(struct, n=neighbor_index)
        for item in neighbor_nn_info:
            neighbor_all_nn_info.append(item)

    for atom in neighbor_all_nn_info:
        for f_atom in nn_info:
          try:
            if operator.eq(f_atom, atom):
                    neighbor_all_nn_info.remove(atom)
            if atom['site_index'] == H_index:
                neighbor_all_nn_info.remove(atom)
          except:
              continue
    # Find the nextnearestcoordination
    next_coord, next_positions = __get_coordination_string(neighbor_all_nn_info)
    neighbor_all_nn_info2 =[]
    for neighbor_info in neighbor_all_nn_info:
        neighbor_index = neighbor_info['site_index']
        neighbor_nn_info = vnn_loose.get_nn_info(struct, n=neighbor_index)
        for item in neighbor_nn_info:
            neighbor_all_nn_info2.append(item)

    for atom in neighbor_all_nn_info2:
        for f_atom in neighbor_all_nn_info:
          try:
            if operator.eq(f_atom, atom):
                    neighbor_all_nn_info2.remove(atom)
            if atom['site_index'] == H_index:
                neighbor_all_nn_info2.remove(atom)
          except:
              continue
    # Find the nextnearestcoordination
    third_coord, third_positions = __get_coordination_string(neighbor_all_nn_info2)

    return {'H_position':H_position,
            'coordination': coordination,
            'coordination_position': coord_position,
            'next_coord': next_coord,
            'next_positions': next_positions,
            'third_coord': third_coord,
            'third_positions': third_positions}
# If we get some QHull or ValueError, then just assume that the adsorbate desorbed



def _concatenate_shell(doc):
    # input:
    #     doc: fingerprint_adslab
    shells = []
    shell_atoms = doc['coordination'].split('-')
    # Sometimes there is no coordination. If this happens, then hackily reformat it
    if shell_atoms == ['']:
        shell_atoms = []
    atom_tuple = set(shell_atoms)
    shell_1_atom = dict(Counter(shell_atoms))
    n_shell_1 = len(shell_1_atom)
    shell_1 = {}
    for atom in atom_tuple:
        distance_set = []
        for item in doc['coordination_position']:
            if item['element'] == atom:
                distance_set.append({'x':item['z'],
                                     'y':item['y'],
                                     'z':item['x']})
        shell_1[atom]= {'number':shell_1_atom[atom],
                        'distanceSet': distance_set}


    # Arg:
    #     doc     A dictionary with the 'neighborcoord' string, whose contents
    #             should look like:
    #                 ['Cu:Cu-Cu-Cu-Cu-Cu-Al',
    #                  'Al:Cu-Cu-Cu-Cu-Cu-Cu']
    # Returns:
    #     second_shell_atoms  An extended list of the coordinations of all
    #                         binding atoms. Continiuing from the example
    #                         shown in the description for the `doc` argument,
    #                         we would get:
    #                         ['Cu', 'Cu', 'Cu', 'Cu', 'Cu', Al, 'Cu', 'Cu', 'Cu', 'Cu', 'Cu', 'Cu']
    # '''

    second_shells = []
    second_shell_atoms = doc['next_coord'].split('-')
    # Sometimes there is no coordination. If this happens, then hackily reformat it
    if second_shell_atoms == ['']:
        second_shell_atoms = []
    second_shell_atoms = list(filter(('H').__ne__, second_shell_atoms))
    second_shells = dict(Counter(second_shell_atoms))
    n_shell_2 = len(second_shells)
    atom2_tuple = set(second_shell_atoms)
    shell_2 = {}
    for atom in atom2_tuple:
        distance_set = []
        for item in doc['next_positions']:
            if item['element'] == atom:
                distance_set.append({'x':item['z'],
                                     'y':item['y'],
                                     'z':item['x']})
        shell_2[atom]= {'number':second_shells[atom],
                        'distanceSet': distance_set}

    third_shells = []
    third_shell_atoms = doc['third_coord'].split('-')
    # Sometimes there is no coordination. If this happens, then hackily reformat it
    if third_shell_atoms == ['']:
        third_shell_atoms = []
    third_shell_atoms = list(filter(('H').__ne__, third_shell_atoms))
    third_shells = dict(Counter(third_shell_atoms))
    n_shell_3 = len(third_shells)
    atom3_tuple = set(third_shell_atoms)
    shell_3 = {}
    for atom in atom3_tuple:
        distance_set = []
        for item in doc['third_positions']:
            if item['element'] == atom:
                distance_set.append({'x':item['z'],
                                     'y':item['y'],
                                     'z':item['x']})
        shell_3[atom]= {'number':third_shells[atom],
                        'distanceSet': distance_set}
    return {'H_position':doc['H_position'],
            'first_layer': shell_1,
            'n_1_layer': n_shell_1,
            'second_layer': shell_2,
            'n_2_layer': n_shell_2,
            'third_layer': shell_3,
            'n_3_layer': n_shell_3}


def fingerprint_element_attributes(elementStr):
    # attributs:
    #  element: str of a element, eg 'H'
    #  tricky: pickle loaded 'tricky_attributes.pkl'
    #  type of method you want to apply. string. Uliss, Han_fyp, Han_2022

    # from mendeleev
    ele = element(elementStr)
    ato_num = ele.atomic_number
    valence_electron = ele.nvalence()
    EN_Allen = ele.en_allen
    EN_Pauling = ele.en_pauling
    EN_Ghosh = ele.en_ghosh
    radias = ele.atomic_radius
    weight = ele.atomic_weight
    volume = ele.atomic_volume
    electron_afiinity = ele.electron_affinity
    vdw_radias = ele.vdw_radius
    elemental_prop = {'nvalence': valence_electron,
                      'atomic_number': ato_num,
                      'atomic_radias': radias,
                      'atomic_weight': weight,
                      'atomic_volume': volume,
                      'vdw_radias': vdw_radias,
                      'elec_neg_Pauling': EN_Pauling,
                      'elec_neg_Allen': EN_Allen,
                      'elec_neg_Ghosh': EN_Ghosh,
                      'electron_affinity': electron_afiinity
                      }

    # from pymatgen
    elementBase = pmpt.Element(elementStr)
    try:
        ele_res = float(elementBase.__getattr__('electrical_resistivity')['value'])
    except:
        print(elementStr + ' electrical_resistivity' + ' is unaviliable')
        ele_res = 0
    try:
        thermal_con = float(elementBase.__getattr__('thermal_conductivity')['value'])
    except:
        print(elementStr + ' thermal_conductivity' + ' is unaviliable')
        thermal_con = 0
    try:
        miner_hard = elementBase.__getattr__('mineral_hardness')
    except:
        print(elementStr + ' mineral_hardness' + ' is unaviliable')
        miner_hard = 0
    try:
        melting = float(elementBase.__getattr__('melting_point')['value'])
    except:
        print(elementStr + ' melting_point' + ' is unaviliable')
        melting = 0
    try:
        boiling = float(elementBase.__getattr__('boiling_point')['value'])
    except:
        print(elementStr + ' boiling_point' + ' is unaviliable')
        boiling = 0

    macro_prop = {'electrical_resistivity': ele_res,
                  'thermal_conductivity': thermal_con,
                  'melting_point': melting,
                  'boiling_point': boiling,
                  'mineral_hardness': miner_hard}

    return {'elemental properties': elemental_prop,
            'macro properties': macro_prop}


def blank_fingerprint(doc, tricky, elemental_attri, type):
    # this function is used to create blank fingerprint (dummy fingerprint) to fill positions
    # parameters:
    #   doc: VT2
    #   tricky: tricky_attributes
    #   elemental attributes(gained before)
    #   type is string, ULISSI , HAN_FYP, HAN_2022
    if type == 'ULISSI':
        atomic_N_array = []
        NG_array = []
        energy_array = []
        for element in tricky['all_layers']:
            energy_array.append(tricky['all_layers'][element]['case_median'])
        for element in elemental_attri:
            atomic_N_array.append(elemental_attri[element]['elemental properties']['atomic_number'])
            NG_array.append(elemental_attri[element]['elemental properties']['elec_neg_Pauling'])
        return np.array([np.mean(atomic_N_array), 0, np.mean(NG_array), np.mean(energy_array)])

    elif type == 'HAN_FYP':
        atomic_N_array = []
        NG_array = []
        energy_array = []
        for element in doc['first_layer']:
            energy_array.append(tricky['first_layer'][element]['aver_case'])
            atomic_N_array.append(elemental_attri[element]['elemental properties']['atomic_number'])
            NG_array.append(elemental_attri[element]['elemental properties']['elec_neg_Pauling'])
        f_layer = [np.mean(atomic_N_array), 0, np.mean(NG_array), np.mean(energy_array)]
        atomic_N_array = []
        NG_array = []
        energy_array = []
        for element in doc['second_layer']:
            energy_array.append(tricky['second_layer'][element]['aver_case'])
            atomic_N_array.append(elemental_attri[element]['elemental properties']['atomic_number'])
            NG_array.append(elemental_attri[element]['elemental properties']['elec_neg_Pauling'])
        s_layer = [np.mean(atomic_N_array), 0, np.mean(NG_array), np.mean(energy_array)]
        return {'f_blk_fp': np.array(f_layer),
                's_blk_fp': np.array(s_layer)}

    elif type == 'latest':
        atomic_N_array = []
        NG_array = []
        energy_array = []
        distance_array = []
        valence_array = []
        for element in doc['first_layer']:
            energy_array.append(tricky['first_layer'][element]['aver_case'])
            atomic_N_array.append(elemental_attri[element]['elemental properties']['atomic_number'])
            NG_array.append(elemental_attri[element]['elemental properties']['elec_neg_Pauling'])
            distance_array.append(np.mean(calculate_distance(doc['H_position'], doc['first_layer'][element]['distanceSet'])))
            valence_array.append(elemental_attri[element]['elemental properties']['nvalence'])
        f_layer = [np.mean(atomic_N_array), 0, np.mean(NG_array), np.mean(energy_array),np.mean(distance_array), np.mean(valence_array)]
        atomic_N_array = []
        NG_array = []
        energy_array = []
        distance_array = []
        valence_array = []
        for element in doc['second_layer']:
            energy_array.append(tricky['second_layer'][element]['aver_case'])
            atomic_N_array.append(elemental_attri[element]['elemental properties']['atomic_number'])
            NG_array.append(elemental_attri[element]['elemental properties']['elec_neg_Pauling'])
            distance_array.append(np.mean(calculate_distance(doc['H_position'], doc['second_layer'][element]['distanceSet'])))
            valence_array.append(elemental_attri[element]['elemental properties']['nvalence'])
        s_layer = [np.mean(atomic_N_array), 0, np.mean(NG_array), np.mean(energy_array), np.mean(distance_array), np.mean(valence_array)]
        return {'f_blk_fp': np.array(f_layer),
                's_blk_fp': np.array(s_layer)}

    elif type == 'latest2':
        atomic_N_array = []
        NG_array = []
        energy_array = []
        distance_array = []
        valence_array = []
        for element in doc['first_layer']:
            energy_array.append(tricky['first_layer'][element]['aver_case'])
            atomic_N_array.append(elemental_attri[element]['elemental properties']['atomic_number'])
            NG_array.append(elemental_attri[element]['elemental properties']['elec_neg_Pauling'])
            distance_array.append(np.mean(calculate_distance(doc['H_position'], doc['first_layer'][element]['distanceSet'])))
            valence_array.append(elemental_attri[element]['elemental properties']['nvalence']-1)
        f_layer = [np.mean(atomic_N_array), 0, np.mean(NG_array), np.mean(energy_array),np.mean(distance_array), np.mean(valence_array)]
        atomic_N_array = []
        NG_array = []
        energy_array = []
        distance_array = []
        valence_array = []
        for element in doc['second_layer']:
            energy_array.append(tricky['second_layer'][element]['aver_case'])
            atomic_N_array.append(elemental_attri[element]['elemental properties']['atomic_number'])
            NG_array.append(elemental_attri[element]['elemental properties']['elec_neg_Pauling'])
            distance_array.append(np.mean(calculate_distance(doc['H_position'], doc['second_layer'][element]['distanceSet'])))
            valence_array.append(elemental_attri[element]['elemental properties']['nvalence']-1)
        s_layer = [np.mean(atomic_N_array), 0, np.mean(NG_array), np.mean(energy_array), np.mean(distance_array), np.mean(valence_array)]
        atomic_N_array = []
        NG_array = []
        energy_array = []
        distance_array = []
        valence_array = []
        for element in doc['third_layer']:
            energy_array.append(tricky['third_layer'][element]['aver_case'])
            atomic_N_array.append(elemental_attri[element]['elemental properties']['atomic_number'])
            NG_array.append(elemental_attri[element]['elemental properties']['elec_neg_Pauling'])
            distance_array.append(np.mean(calculate_distance(doc['H_position'], doc['third_layer'][element]['distanceSet'])))
            valence_array.append(elemental_attri[element]['elemental properties']['nvalence']-1)
        t_layer = [np.mean(atomic_N_array), 0, np.mean(NG_array), np.mean(energy_array), np.mean(distance_array), np.mean(valence_array)]
        return {'f_blk_fp': np.array(f_layer),
                's_blk_fp': np.array(s_layer),
                't_blk_fp': np.array(t_layer)}

    else:
        print('warining! type name is wrong!')
        return [0, 0, 0, 0]


def normal_fingerprint(element, layer, tricky, elemental_attri, type):

    if type == 'ULISSI':
        atomic_n = elemental_attri[element]['elemental properties']['atomic_number']
        NG_P = elemental_attri[element]['elemental properties']['elec_neg_Pauling']
        median_energy = tricky['all_layers'][element]['case_median']
        return np.array([atomic_n, 0, NG_P, median_energy])

    if type == 'HAN_FYP':
        atomic_n = elemental_attri[element]['elemental properties']['atomic_number']
        NG_P = elemental_attri[element]['elemental properties']['elec_neg_Pauling']
        if layer == 1:
            mean_energy = tricky['first_layer'][element]['case_median']
        elif layer == 2:
            mean_energy = tricky['second_layer'][element]['case_median']
        else:
            print('layer num is not indicated!')
            mean_energy = 0
        return np.array([atomic_n, 0, NG_P, mean_energy])

    if type == 'latest':
        atomic_n = elemental_attri[element]['elemental properties']['atomic_number']
        NG_P = elemental_attri[element]['elemental properties']['elec_neg_Pauling']
        nvalence = elemental_attri[element]['elemental properties']['nvalence']
        if layer == 1:
            mean_energy = tricky['first_layer'][element]['case_median']
        elif layer == 2:
            mean_energy = tricky['second_layer'][element]['case_median']
        else:
            print('layer num is not indicated!')
            mean_energy = 0
        return np.array([atomic_n, 0, NG_P, mean_energy, 0, nvalence])

    if type == 'latest2':
        atomic_n = elemental_attri[element]['elemental properties']['atomic_number']
        NG_P = elemental_attri[element]['elemental properties']['elec_neg_Pauling']
        nvalence = elemental_attri[element]['elemental properties']['nvalence']
        if layer == 1:
            mean_energy = tricky['first_layer'][element]['case_median']
        elif layer == 2:
            mean_energy = tricky['second_layer'][element]['case_median']
        elif layer == 3:
            mean_energy = tricky['third_layer'][element]['case_median']
        else:
            print('layer num is not indicated!')
            mean_energy = 0
        return np.array([atomic_n, 0, NG_P, mean_energy, 0, nvalence-1])

def normal_fingerprint2(element, layer, tricky, elemental_attri, type):


    if type == 'latest2':
        atomic_n = elemental_attri[element]['elemental properties']['atomic_number']
        NG_P = elemental_attri[element]['elemental properties']['elec_neg_Pauling']
        nvalence = elemental_attri[element]['elemental properties']['nvalence']
        if layer == 1:
            mean_energy = tricky['first_layer'][element]['case_median']
        elif layer == 2:
            mean_energy = tricky['second_layer'][element]['case_median']
        elif layer == 3:
            mean_energy = tricky['third_layer'][element]['case_median']
        else:
            print('layer num is not indicated!')
            mean_energy = 0
        mean_energy=mean_energy-0.244
        return np.array([atomic_n, 0, NG_P, mean_energy, 0, nvalence-1])

def final_case_fingerprint(doc, tricky, elemental_attri, type):
    fingerprints = np.array([])
    if type == 'latest2':
        blank_fp = blank_fingerprint(doc, tricky, elemental_attri, type)
        for element in doc['first_layer']:
            normal_fp1 = normal_fingerprint2(element, 1, tricky, elemental_attri, type)
            normal_fp1[1] = doc['first_layer'][element]['number']
            normal_fp1[4] = np.mean(calculate_distance(doc['H_position'], doc['first_layer'][element]['distanceSet']))
            fingerprints = np.append(fingerprints, normal_fp1)
        for i in range(3 - len(doc['first_layer'])):
            fingerprints = np.append(fingerprints, blank_fp['f_blk_fp'])
        #print(normal_fp1[4])
        for element in doc['second_layer']:
            normal_fp2 = normal_fingerprint2(element, 2, tricky, elemental_attri, type)
            normal_fp2[1] = doc['second_layer'][element]['number']
            normal_fp2[4] = np.mean(calculate_distance(doc['H_position'], doc['second_layer'][element]['distanceSet']))
            fingerprints = np.append(fingerprints, normal_fp2)
        for i in range(4 - len(doc['second_layer'])):
            fingerprints = np.append(fingerprints, blank_fp['s_blk_fp'])
        #print(normal_fp2[4])
        for element in doc['third_layer']:
            normal_fp3 = normal_fingerprint2(element, 3, tricky, elemental_attri, type)
            normal_fp3[1] = doc['third_layer'][element]['number']
            normal_fp3[4] = np.mean(calculate_distance(doc['H_position'], doc['third_layer'][element]['distanceSet']))
            fingerprints = np.append(fingerprints, normal_fp3)
        for i in range(4 - len(doc['third_layer'])):
            fingerprints = np.append(fingerprints, blank_fp['t_blk_fp'])

    return fingerprints

def make_label(dir, trainIndex, data, tricky, element_properties, type):

    result = pd.read_csv('./id_prop.csv')
    FP = []
    results = []
    index_array = []
    
    for i in range(len(result)):
        index_array.append(result['name'][i])


    print(index_array)
    for index in tqdm(index_array):
        for doc in data:
            if str(doc['index']) == str(index):
                fp = final_case_fingerprint(doc['VT2'], tricky, element_properties, type).tolist()
                FP.append(fp)
                results.append([doc['energy']])
    return {'FP': np.array(FP),
            'result': np.array(results)}