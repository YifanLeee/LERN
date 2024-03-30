# LERN:Local environment interaction-based machine learning framework for predicting molecular adsorption energy
This software package implements the Local environment ResNet (LERN) that takes an framework for predicting molecular adsorption energy. 

The package provides two major functions:

- Train a LERN model with a customized dataset.
- Predict material properties of new molecular with a pre-trained LERN model.

The following paper describes the details of the LEI-framework:

[Local environment interaction-based machine learning framework for predicting molecular adsorption energy](https://www.oaepublish.com/pre_onlines/jmi.2023.41)

## Table of Contents

- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Define a customized dataset](#define-a-customized-dataset)
  - [Train a LERN model](#train-a-lern-model)
  - [Predict material properties with a pre-trained LERN model](#predict-molecular-properties-with-a-pre-trained-lern-model)
- [Authors](#authors)
- [License](#license)

## How to cite

Please cite the following work if you want to use LERN.

```
Li Y, Wu Y, Han Y, Lyu Q, Wu H, Zhang X, Shen L. Local environment interaction-based machine learning framework for predicting molecular adsorption energy. J Mater Inf 2024;4:[Accept]. http://dx.doi.org/10.20517/jmi.2023.41
```

##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](https://scikit-learn.org/)
- [ase](https://wiki.fysik.dtu.dk/ase/)
- [pymatgen](http://pymatgen.org)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following command to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named `lern` and install all prerequisites:

```
conda upgrade lern
conda create -n lern python=3.9 pytorch torchvision ase pymatgen -c pytorch -c conda-forge
```

This creates a conda environment for running LERN. Before using LERN, activate the environment by:

```
conda activate lern
```

## Usage

### Define a customized dataset 

To input Local Environment to LERN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the moleculars that you are interested in
- The target properties for each molecular (not needed for predicting, but you need to put some random numbers(eg."0") in `id_prop.csv`)

You can create a customized dataset by creating the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.)

3. `cif` folder: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the molecular structure, where file name is the unique `name` for the molecular.

The structure of the dataset should be:

```
id_prop.csv
cif
├── name0.cif
├── name1.cif
├── ...
```

### Train a LERN model

Before training a new LERN model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) to store the structure-property relations of interest.

Then, in directory `lern`, you can train a LERN model for your customized dataset by:

```
python main.py
```

After training, you will get two files in `lern` directory.

- `model.pth`: stores the LERN model with the best validation accuracy.
- `results.csv`: stores the `name`, target value, and predicted value for each molecular.

### Predict material properties with a pre-trained LERN model

Before predicting the material properties, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) for all the crystal structures that you want to predict.
- Obtain a [pre-trained LERN model](pre-trained) named `model.pth`.

Then, in directory `lern`, you can predict the properties of the moleculars:

```
python predict.py
```

After predicting, you will get one file in `lern` directory:

- `predict.csv`: stores the `name`and predicted value for each molecular. 


## Authors

This software was primarily written by [Li Yifan](https://scholar.google.com/citations?hl=zh-CN&user=NuZDso4AAAAJ&view_op=list_works&gmla=AKKJWFe62012wCjalf5_D-_q3CsSLMd-Ob5pQXSjCDF5ZHtDGwndPsghdUI6om2v9DxsGvReQ2-am-F5qNseTX0W-VeaOwkIHmT-gii6GiiTaoIQb91OtXZ3mUu1blo9mfECMbHBX9X2q2nn4dN4ck6z-65ASpNd9FL9n9ItVXTYwgtz_2HnZGN1O6E5xQ&sciund=4912371730478261254) who was advised by [Prof. Shen Lei](https://cde.nus.edu.sg/me/staff/shen-lei/). 

## License

LERN is released under the NUS License.
