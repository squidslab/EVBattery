# EVBattery – Battery Capacity Estimation

Repository for the analysis and preprocessing of datasets on the health status of electric vehicle batteries, specifically for **battery capacity estimation**.

This project is based on and builds upon the work presented by He et al. in their paper; more information in [*References*](#references-and-license). 

For the sake of readers, a brief summary of the paper is also provided in this README when needed.


---

## Table of contents

- [Project goal](#project-goal)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Environments](#environments)
- [Installation from scratch](#installation-from-scratch)
  - [1. Install Miniconda](#1-install-miniconda)
  - [2. Create the Python 3.6 environment](#2-create-the-python-36-environment)
  - [3. Install the dependencies](#3-install-the-dependencies)
- [Generating the five-fold files](#generating-the-five-fold-files)
- [Understanding `five_fold_utils`](#understanding-five_fold_utils) 
- [How the code loads and splits the data](#how-the-code-loads-and-splits-the-data)
- [Running the capacity-estimation code](#running-the-capacity-estimation-code)
- [Main models](#main-models)
- [Expected computational cost](#expected-computational-cost)
- [References and license](#references-and-license)

---

# Project goal

The main objective of this project is to study the EVBattery dataset and establish a reproducible baseline which requires less datas for **battery capacity estimation**.

The authors formulate capacity estimation as a regression problem. Only charging snippets for which a capacity value is available are used in the capacity-estimation experiments.

The authors benchmark:
- Random Forest
- XGBoost
- MLP
- Gated CNN
- LSTM

The current implementation contains all of these branches in `capacity_estimation/main.py`. We will focus on Random Forest, XGBoost and LSTM later in this README.

For the thesis work, the authors' implementation is treated as a **black box**: the first objective is to reproduce and understand the original pipeline without making independent modifications.



# Dataset

## EVBattery dataset

The EVBattery dataset was collected from real-world electric vehicles from three manufacturers. The public release contains:

| Dataset | Vehicles | Anomaly vehicles | Charging snippets | Capacity labels |
|---|---:|---:|---:|---:|
| `battery_dataset1` | 217 | 31 | 629,121 | 349,741 |
| `battery_dataset2` | 198 | 1 | 472,829 | 203,207 |
| `battery_dataset3` | 49 | 16 | 176,327 | 32,974 |
| Total | 464 | 48 | 1,278,277 | 585,922

The paper states that the capacity labels are real values in approximately the range **28.28–46.23 Ah**.

The complete dataset contains more than 1.2 million charging snippets and working with all these data isn't trivial, as we will see later.

### Features

Each charging snippet contains 128 observations for eight time-series features:

1. average cell voltage
2. charging current
3. SOC
4. maximum cell voltage
5. minimum cell voltage
6. maximum cell temperature
7. minimum cell temperature
8. timestamp

The metadata associated with a snippet includes information such as vehicle number, mileage, charge-segment index, health label and capacity.

The raw public data is stored as pickle (`.pkl`) files.

---
To download the .zip with these datasets, see [References](#references-and-license).


# Repository structure

The expected project layout is:

```text
EVBattery/
│
├── original/
│   ├── battery_dataset1/
│   │   ├── data/
│   │   │   ├── *.pkl
│   │   │   └── ...
│   │   └── label/
│   │       └── label.csv
│   │
│   ├── battery_dataset2/
│   │   └── ...
│   │
│   └── battery_dataset3/
│       └── ...
│
├── processed/
│   ├── dataset1/
│   │   ├── data/
│   │   │   ├── *.dat
│   │   │   └── ...
│   │   └── metadata.pkl
│   │
│   ├── dataset2/
│   │   └── ...
│   │
│   └── dataset3/
│       └── ...
│
└── scripts/
    ├── authors_code/
    │   └── battery_dataset_neurips23dataset_code/
    │       │
    │       ├── capacity_estimation/
    │       │   ├── capacity_dataset.py
    │       │   ├── main.py
    │       │   └── ...
    │       │
    │       ├── five_fold_utils/
    │       │   ├── all_car_dict.npz.npy
    │       │   ├── ind_odd_dict.npz.npy
    │       │   ├── ind_odd_dict1.npz.npy
    │       │   ├── ind_odd_dict2.npz.npy
    │       │   └── ind_odd_dict3.npz.npy
    │       │
    │       └── ...
    │
    └── my_code/
        ├── analyze_datasets.py
        ├── consolidate_datasets.py
        ├── inspect_pkl.py
        └── ...

```

> **Important:** the relative paths are significant for both authors' code and my code.

The `processed/` dir will be automatically created by **`consolidate_datasets.py`**. You can check what each script in `my_code/` does by opening the corresponding file: the first few lines provide an overview of its main purpose.


# Environments

This project uses two separate Python environments:

The scripts developed for this project, located in `my_code/`, use:

- Python 3.11
- NumPy 2.4.6

The original code provided by the authors,located in `authors_code/`, requires a separate conda environment based on Python 3.6. The dependecies of their code are listed in the next section.

> **Important:** The two environments should be kept separate because they require different Python and package versions.



# Installation from scratch

The following procedure recreates the environment used for the capacity-estimation experiments.

## 1. Install Miniconda

Download and install Miniconda for Linux:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run the installer:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the installer instructions and restart the shell, or load Conda manually if requested by the installer.

Verify:

```bash
conda --version
```

---

## 2. Create the Python 3.6 environment

The authors' code is old and was released for Python 3.6.

Create the environment:

```bash
conda create -n evbattery python=3.6
```

Activate it:

```bash
conda activate evbattery
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.6.xx
```

Also verify which Python executable is being used:

```bash
which python
```

Expected path is similar to:

```text
/home/<user>/miniconda3/envs/evbattery/bin/python
```

---

## 3. Install the dependencies

### Core dependencies used by `my_code`

As it's used just numpy, you can simply write with your env activated:

```bash
pip install numpy
```
---
### Core dependencies used by `capacity_estimation`

You can install the exact versions used during with:

```bash
pip install -r requirements.txt
```
> **Important**: Make sure that the evbattery conda environment is activated before installing the requirements. Otherwise, the packages may be installed into your system Python environment or another active environment.

I do not recommend doing the same with the requirements of the original authors' code, as they include also dependencies that are only needed for the battery health anomaly detection task, which are not covered in this repository.

---

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If it prints *False*, you have to use the CPU-only setup of this project. Otherwise, you can uncomment the `.cuda()` calls in the **`main.py`**, in order to improve the performance.


# Generating the five-fold files

Before running `capacity_estimation/main.py`, the authors' preprocessing code must generate the files used to organize the vehicles and perform the five-fold cross-validation split.

The important files are:

```text
five_fold_utils/
├── all_car_dict.npz.npy
├── ind_odd_dict.npz.npy
├── ind_odd_dict1.npz.npy
├── ind_odd_dict2.npz.npy
└── ind_odd_dict3.npz.npy
```

These files do not contain the actual charging-snippet arrays.

Instead, they contain indexing and split information used later by `CapacityDataset`.

The actual charging snippets remain stored in the original `.pkl` files.

If these files do not exist, `main.py` will fail when it tries to load any of them.


# Understanding `five_fold_utils`

## `all_car_dict.npz.npy`

This dictionary maps each vehicle number to the list of pickle files belonging to that vehicle.

During preprocessing, the code reads every `.pkl` file and extracts the vehicle number from its metadata.

The dictionary is then built appending each path to the vehicle number.

Conceptually, the resulting structure is:

```text
dict
 ├── [vehicle number X]
 │     ├── PKL path
 │     ├── PKL path
 │     └── ...
 ├── [vehicle number Y]
 │     ├── PKL path
 │     └── ...
 └── ...
```

For example:

```python
all_car_dict[512]
```

returns the list of `.pkl` files belonging to vehicle `512`.

The global `all_car_dict` contains 464 vehicle IDs, corresponding to the three EVBattery datasets combined.

Therefore, `all_car_dict` is an index from:

```text
vehicle ID → PKL file paths
```

It does not contain the actual `128 × 8` charging-snippet arrays.

## `ind_odd_dict*.npz.npy`

These dictionaries contain the vehicle-number lists used for the cross-validation split.

The terminology comes from the authors' code:

```text
ind → in-distribution
ood → out-of-distribution
```

The preprocessing code first separates vehicles according to the label stored in the PKL metadata (corresponding to the anomaly label):

```python
if this_pkl_file[1]['label'] == '00':
    ind_car_num_list1.add(this_car_number)
else:
    ood_car_num_list1.add(this_car_number)
```

The resulting sets contain unique vehicle numbers rather than individual PKL files.

The vehicle IDs are then converted to lists, sorted, and shuffled:

```python
random.seed(0)

ind_sorted = sorted(ind_car_num_list1)
random.shuffle(ind_sorted)

ood_sorted = sorted(ood_car_num_list1)
random.shuffle(ood_sorted)
```

`random.seed(0)` initializes Python's pseudo-random number generator with a fixed seed. This makes the shuffle reproducible: running the same preprocessing code again produces the same vehicle ordering.

`sorted()` provides a deterministic starting order before the random shuffle.

The resulting lists are stored as:

```python
ind_odd_dict = {}

ind_odd_dict["ind_sorted"] = ind_sorted
ind_odd_dict["ood_sorted"] = ood_sorted
```

and saved with:

```python
np.save('../five_fold_utils/ind_odd_dict1.npz', ind_odd_dict)
```

The files are therefore **vehicle-level split information**, not the actual battery snippets.

The files have the following roles:

```text
ind_odd_dict1.npz.npy
    → vehicles from dataset 1

ind_odd_dict2.npz.npy
    → vehicles from dataset 2

ind_odd_dict3.npz.npy
    → vehicles from dataset 3

ind_odd_dict.npz.npy
    → vehicles from all three datasets combined
```

The global lists are created by combining the three dataset-specific lists.


# How the code loads and splits the data

The capacity pipeline performs the cross-validation split at the **vehicle level**, not at the individual charging-snippet level.

`CapacityDataset` loads:

```text
all_car_dict.npz.npy
ind_odd_dict*.npz.npy
```

and obtains:

```python
self.ind_car_num_list = ind_ood_car_dict['ind_sorted']
self.ood_car_num_list = ind_ood_car_dict['ood_sorted']
```

For a given fold, one fifth of the IND vehicles and one fifth of the OOD vehicles are assigned to the test set. Conceptually:

```text
IND vehicles
    │
    ├── 4/5 → training
    └── 1/5 → testing

OOD vehicles
    │
    ├── 4/5 → training
    └── 1/5 → testing
```

The two resulting groups are then combined to create the training or test vehicle list.

For the training dataset, the selected fold is excluded:

```python
car_number = (
    self.ind_car_num_list[:int(fold_num * len(self.ind_car_num_list) / 5)]
    + self.ind_car_num_list[int((fold_num + 1) * len(self.ind_car_num_list) / 5):]
    + self.ood_car_num_list[:int(fold_num * len(self.ood_car_num_list) / 5)]
    + self.ood_car_num_list[int((fold_num + 1) * len(self.ood_car_num_list) / 5):]
)
```

For the test dataset, only the selected fold is used:

```python
car_number = (
    self.ind_car_num_list[
        int(fold_num * len(self.ind_car_num_list) / 5):
        int((fold_num + 1) * len(self.ind_car_num_list) / 5)
    ]
    +
    self.ood_car_num_list[
        int(fold_num * len(self.ood_car_num_list) / 5):
        int((fold_num + 1) * len(self.ood_car_num_list) / 5)
    ]
)
```

Once the vehicle numbers have been selected, `CapacityDataset` uses `all_car_dict` to find the corresponding PKL files:

```python
for each_num in car_number:
    for each_pkl in self.all_car_dict[each_num]:
        train1 = torch.load(each_pkl)
```

Only snippets whose capacity is different from zero are then added to the dataset used for the estimation problem:

```python
if train1[1]["capacity"] != 0:
    self.battery_dataset.append(train1)
```

Therefore, the overall process is:

```text
vehicle IDs
    │
    ▼
fold selection
    │
    ▼
all_car_dict
    │
    ▼
PKL file paths
    │
    ▼
torch.load()
    │
    ▼
charging snippet
    │
    ├── data: 128 × 8
    └── metadata: vehicle, label, capacity, etc.
    │
    ▼
capacity != 0
    │
    ▼
battery_dataset.append()
```


# Running the capacity-estimation code

Move into the capacity-estimation directory:

```bash
cd ~/University/Tesi/EVBattery/scripts/authors_code/battery_dataset_neurips23dataset_code/capacity_estimation
```

If the repository was cloned somewhere else, use the corresponding absolute/relative path.

Make sure the Conda environment is active:

```bash
conda activate evbattery
```

Then run the default LSTM experiment:

```bash
python main.py
```

## Command-line arguments

The default arguments in the current `main.py` are:

```text
fold_num   = 0
batch_size = 64
model      = LSTMNet
num_epochs = 10
```

Therefore:

```bash
python main.py
```

is equivalent to:

```bash
python main.py \
    --fold_num 0 \
    --batch_size 64 \
    --model LSTMNet \
    --num_epochs 10
```

### `--fold_num`

Selects the cross-validation fold. \
Valid values for five-fold cross-validation are: 0, 1, 2, 3 or 4.

```bash
python main.py --fold_num 4
```

### `--batch_size`

Determines how many samples are processed at once during training. After processing one batch, the model uses the computed error to update its parameters.

```bash
python main.py --batch_size 32
```

### `--model`

Selects the model between:

- LSTMNet
- MLP
- GatedCNN
- XGBoost
- RandomForest
- MEAN

```bash
python main.py --model XGBoost
```

### `--num_epochs`

For the LSTM/MLP/GatedCNN implementations, this is the number of training epochs. \
For XGBoost, it is equivalent to the number of boosting rounds.

```bash
python main.py --num_epochs 50
```

### `--load_saved_dataset`

If present, the script loads previously serialized datasets from `saved_dataset/` instead of rebuilding them.

```bash
python main.py --load_saved_dataset
```


# Main models

Some of the available models are the following:

### 1. LSTM

The paper describes the LSTM architecture as an LSTM layer followed by two fully connected layers with ReLU activation, hidden dimension 32, Adam with learning rate `0.001`, and 10 training epochs.

The implementation uses:

```text
input_dim = 8
hidden_dim = 32
output_dim = 1
learning_rate = 0.001
loss = MSELoss
```
---

### 2. XGBoost

The current implementation uses:

```text
objective   = reg:squarederror
eta         = 0.1
max_depth   = 4
eval_metric = rmse
```

The number of boosting rounds is taken from `--num_epochs` and should be 50.

---

### 3. Random Forest

The current implementation uses:

```text
n_estimators = 10
random_state = 0
n_jobs       = 10
max_depth    = 4
```


# Expected computational cost

The main bottleneck is data loading and preprocessing.

The code iterates through many vehicle-associated `.pkl` files and filters them according to the capacity label.

On the CPU-only Ryzen 5 3500U machine used during this project, a single fold can take several minutes just to load/process the data before the actual training is complete.

Neural-network training is also significantly slower on CPU than on the NVIDIA GPUs used in the authors' experiments.

## Very large memory usage

The capacity loader builds:

```python
self.battery_dataset = []
```

and appends every capacity-labeled snippet to the in-memory dataset. The dataset is therefore not streamed one file at a time during training. This can require **substantial RAM**.

On systems with limited RAM, the process may become very slow or the system may run out of memory while loading and preprocessing the data.

If you experience memory-related issues, monitor the system's RAM usage during execution and make sure that sufficient memory is available before running the full pipeline.


# References and license

The project is based on the following paper:

> Haowei He et al., *EVBattery: A Large-Scale Electric Vehicle Dataset for Battery Health and Capacity Estimation*, arXiv:2201.12358v3, 2023.

You can read the [paper](https://arxiv.org/abs/2201.12358) on arXiv and download the [EVBattery Dataset](https://doi.org/10.6084/m9.figshare.23301881)  on Figshare