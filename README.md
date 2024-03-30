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

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

There are two examples of customized datasets in the repository: `data/sample-regression` for regression and `data/sample-classification` for classification. 

**For advanced PyTorch users**

The above method of creating a customized dataset uses the `CIFData` class in `cgcnn.data`. If you want a more flexible way to input crystal structures, PyTorch has a great [Tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#sphx-glr-beginner-data-loading-tutorial-py) for writing your own dataset class.

### Train a CGCNN model

Before training a new CGCNN model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

Then, in directory `cgcnn`, you can train a CGCNN model for your customized dataset by:

```bash
python main.py root_dir
```

You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`. Alternatively, you may use the flags `--train-ratio`, `--val-ratio`, `--test-ratio` instead. Note that the ratio flags cannot be used with the size flags simultaneously. For instance, `data/sample-regression` has 10 data points in total. You can train a model by:

```bash
python main.py --train-size 6 --val-size 2 --test-size 2 data/sample-regression
```
or alternatively
```bash
python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/sample-regression
```

You can also train a classification model with label `--task classification`. For instance, you can use `data/sample-classification` by:

```bash
python main.py --task classification --train-size 5 --val-size 2 --test-size 3 data/sample-classification
```

After training, you will get three files in `cgcnn` directory.

- `model_best.pth.tar`: stores the CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

### Predict material properties with a pre-trained CGCNN model

Before predicting the material properties, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` for all the crystal structures that you want to predict.
- Obtain a [pre-trained CGCNN model](pre-trained) named `pre-trained.pth.tar`.

Then, in directory `cgcnn`, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py pre-trained.pth.tar root_dir
```

For instace, you can predict the formation energies of the crystals in `data/sample-regression`:

```bash
python predict.py pre-trained/formation-energy-per-atom.pth.tar data/sample-regression
```

And you can also predict if the crystals in `data/sample-classification` are metal (1) or semiconductors (0):

```bash
python predict.py pre-trained/semi-metal-classification.pth.tar data/sample-classification
```

Note that for classification, the predicted values in `test_results.csv` is a probability between 0 and 1 that the crystal can be classified as 1 (metal in the above example).

After predicting, you will get one file in `cgcnn` directory:

- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set. Here the target value is just any number that you set while defining the dataset in `id_prop.csv`, which is not important.


## Authors

This software was primarily written by [Li Yifan] who was advised by [Prof. Shen Lei](https://cde.nus.edu.sg/me/staff/shen-lei/). 

## License

CGCNN is released under the NUS License.
