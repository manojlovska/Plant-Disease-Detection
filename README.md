# Detecting plant diseases using ResNet9
## Introduction
The goal of this project is to recognize diseased plants and distinguish between the different types of diseases that can occur. For the purpose of this project the Plant Village Dataset (without augmentation) was used, which can be found [here](https://data.mendeley.com/datasets/tywbtsjrjv/1). This project was created with a little assistance from this [Kaggle tutorial](https://www.kaggle.com/code/atharvaingle/plant-disease-classification-resnet-99-2), although there were quite some changes made to the code, especially in constructing the network. An accuracy of 99.02% was achieved.

## Steps to reproduce
### Step 1: Clone the repository
```shell
git clone https://github.com/manojlovska/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### Step 2: Create virtual environment with conda and activate it
```shell
conda create -n env python=3.8.5
conda activate env
```
### Step 3: Install the requirements
```shell
pip install -r requirements.txt
```
### Step 4: Download the dataset
You can download the dataset from [here](https://data.mendeley.com/datasets/tywbtsjrjv/1). Then, make the directories as suggested bellow and move the dataset in there.
```shell
mkdir datasets
cd datasets
mkdir PlantVillage
cd PlantVillage
```
```shell
mv -r /path/to/Plant_leave_diseases_dataset_without_augmentation/ .
```
### Step 5: Split the data into train and validation sets
```shell
python -m preprocessing.train_test_split
```
Running this script will modify the dataset folder structure from this:
```
.
├── datasets
│   └── PlantVillage
│       ├── Plant_leave_diseases_dataset_without_augmentation
│       │   ├── Apple___Apple_scab
│       │   ├── Apple___Black_rot
│       │   ├── Apple___Cedar_apple_rust
│       │   ├── Apple___healthy
│       │   ├── Background_without_leaves
│       │   ├── Blueberry___healthy
│       │   ├── Cherry___healthy
│       │   ├── Cherry___Powdery_mildew
│       │   ├── Corn___Cercospora_leaf_spot Gray_leaf_spot
│       │   ├── Corn___Common_rust
│       │   ├── Corn___healthy
│       │   ├── Corn___Northern_Leaf_Blight
│       │   ├── Grape___Black_rot
│       │   ├── Grape___Esca_(Black_Measles)
│       │   ├── Grape___healthy
│       │   ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
│       │   ├── Orange___Haunglongbing_(Citrus_greening)
│       │   ├── Peach___Bacterial_spot
│       │   ├── Peach___healthy
│       │   ├── Pepper,_bell___Bacterial_spot
│       │   ├── Pepper,_bell___healthy
│       │   ├── Potato___Early_blight
│       │   ├── Potato___healthy
│       │   ├── Potato___Late_blight
│       │   ├── Raspberry___healthy
│       │   ├── Soybean___healthy
│       │   ├── Squash___Powdery_mildew
│       │   ├── Strawberry___healthy
│       │   ├── Strawberry___Leaf_scorch
│       │   ├── Tomato___Bacterial_spot
│       │   ├── Tomato___Early_blight
│       │   ├── Tomato___healthy
│       │   ├── Tomato___Late_blight
│       │   ├── Tomato___Leaf_Mold
│       │   ├── Tomato___Septoria_leaf_spot
│       │   ├── Tomato___Spider_mites Two-spotted_spider_mite
│       │   ├── Tomato___Target_Spot
│       │   ├── Tomato___Tomato_mosaic_virus
│       │   └── Tomato___Tomato_Yellow_Leaf_Curl_Virus
├── ...
```
to this:
```
.
├── datasets
│   └── PlantVillage
│       └── Plant_leave_diseases_dataset_without_augmentation
│           ├── all_data
│           ├── train
│           └── val
├── ...
```
Note:
* **all_data** directory contains all the images from the dataset
* **train** directory contains training images (80% of all the images)
* **val** directory contains the validation set (20% of all the images)
### Step 6 (Optional): Run the following command for some EDA, to get to know the dataset
```shell
python -m preprocessing.eda
```
Note: Running this script will provide you some useful information of the dataset you will be using. It might be helpful to get to know the data you will deal with before starting the whole process of training the network.
### Step 7: Train the ResNet9 model
To train the ResNet9 neural network run the following command:
```shell
python -m plantdisease.tools.train -f /path/to/plant_disease_conf_class.py -d 1 -b 32 --fp16 
```
* -f: used to specify your configuration file
* -d: number of devices to train on
* -b: batch_size
* -fp16: mixed precision training 


**Logging to Weights & Biases** \
In case you want to implement model performance tracking with [W&B](https://wandb.ai/site), add the command line argument `--logger wandb`, and use the prefix "wandb-" to specify arguments for initializing the wandb run.
```shell
python -m plantdisease.tools.train -f /path/to/plant_disease_conf_class.py -d 1 -b 32 --fp16 --logger wandb wandb-project <project-name>
```
An example of a wandb dashboard can be found [here](https://wandb.ai/team_e7/Plant-Disease-Detection?workspace=user-annastasijamanojlovska).

### Step 8: Model evaluation
```#TODO```