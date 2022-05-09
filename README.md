# Project Information
Format reference: https://github.com/paperswithcode/releasing-research-code

The aim of this project is to understand, replicate and extend the paper the paper ['Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit'](https://arxiv.org/pdf/2007.09483v4.pdf) by Emma Rocheteau, Pietro Li, and
Stephanie Hyland ( [repository](https://github.com/EmmaRocheteau/TPC-LoS-prediction)). The paper focuses on creating a deep learning model that will aid in hospital bed management, a daily challenge for the healtcare industry. 

This repository contains a modified version of the orginal work, done in order to carry out and extend the experiments.
## Requirements
To reproduct this experment, it is necessarily to install Python, to meet some of the requirements listed above, it is necessary to utilize version 3.7 or above. Python can be downloaded and installed directly from [here](https://www.python.org/). 

### Specification of Dependencies
In order to preform our experiements, particular versions of various libraries and tools must be used. The following is the required versions, this information is contained within the file requirements.txt.

```txt
numpy==1.18.1  
pandas==0.24.2  
scipy==1.4.1  
torch==1.5.0  
trixi==0.1.2.2  
scikit-learn==0.20.2  
captum==0.2.0  
shap==0.35.0
```

After successful installation of Python, several libraries must be installed. We conducted our experments on a Windows based machine. The following commands can be used to install the necessary libraries on computers utilizing Windows. 

```shell
pip install numpy==1.18.1  
pip install pandas==0.24.2  
pip install scipy==1.4.1  
pip install torch==1.5.0  
pip install trixi==0.1.2.2  
pip install scikit-learn==0.20.2  
pip install captum==0.2.0  
pip install shap==0.35.0
```

### Download Data
In order to reproduce this the experements the eICU data must be downloaded from the eICU Collaborative Reserach Database (version 2.0). Credentials, earned through completing an online course, are required to access. When credentialized, the data can be access through [Physionet.org]( https://physionet.org/content/eicu-crd/2.0/).

### Local Database Setup eICU Database Locally
To begin, first download and install [Postgres]( http://www.postgresql.org/download/). After a successful installation generation a local databse and open SQL Shell (psql). Enter the following commands to setup the database.
- Generate data tables
```shell
\i [eicu-code path]/postgres/postgres_create_tables.sql
```
- Open data directory and load the data
```shell
\cd [eicu-database path]
\i [eicu-project-code path]/postgres/postgres_load_data_gz.sql
```
-  Navigate to project folder and create the tables
```shell
\cd [project path]/
\i eICU_preprocessing/create_all_tables.sql
```

- Preprocess the data
	- Modify the directory paths in path.json to point to destinations on your machine. At this point the data can be pre-process. 
```shell
python -m eICU_preprocessing.run_all_preprocessing
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :


Below is the result comparison for TPC model between the original paper and our replication experiment on eICU test data.

| **TPC Model**           | **MAD**   | **MSE**   | **MAPE** | **MSLE**  | **R<sup>2</sup>**   | **KAPPA** |
|---------------|-----------|-----------|----------|-----------|-----------|-----------|
| Original Paper | 1.78±0.02 |  21.7±0.5 | 63.5±4.3 | 0.70±0.03 | 0.27±0.02 | 0.58±0.01 |
| Our Results    | 2.496   | 24.628    | 197.249   | 0.328     | 0.475     | 0.710     |

Comparing Models
| **Model**           | **MAD** | **MSE** | **MAPE** | **MSLE** | **R<sup>2</sup>** | **KAPPA** |
| ------------------- | ------- | ------- | -------- | -------- | ----------------- | --------- |
| TPC                 | 2.496   | 24.628  | 197.249  | 0.328    | 0.475             | 0.710     |
| Transformer | 0       | 0       | 0        | 0        | 0                 | 0         |
| LSTM (Channel-wise)  | 0        |  0       | 0         |  0        |  0                 |   0        |


>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Citations


```bibtex
@inproceedings{rocheteau2021,
	author = {Rocheteau, Emma; Li, Pietro; Hyland, Stephanie},
	title = {Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit},
	year = {2021},
	isbn = {9781450383592},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3450439.3451860},
	doi = {10.1145/3450439.3451860},
	booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
	pages = {58–68},
	numpages = {11},
	keywords = {intensive care unit, length of stay, temporal convolution, mortality, patient outcome prediction},
	location = {USA},
	series = {CHIL '21}
}
```

## References

Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series Data. Scientific Data, 6(96), 2019.
