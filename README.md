# Project Information
Format reference: https://github.com/paperswithcode/releasing-research-code

The aim of this project is to understand, replicate and extend the paper ['Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit'](https://arxiv.org/pdf/2007.09483v4.pdf) by Emma Rocheteau, Pietro Li, and
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
In order to reproduce this the experements the eICU data must be downloaded from the MIMIC-IV (Medical Information Mart for Intensive Care) database. The data can be access through [Physionet.org]( https://physionet.org/content/eicu-crd/2.0/).

### Local Database Setup eICU Database Locally
After a successful installation generation a local databse and open SQL Shell (psql). Enter the following commands to setup the database.
- Generate data tables https://eicu-crd.mit.edu/tutorials/install_eicu_locally/ 
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

### Preprocessing
1. To begin, first download and install [Postgres]( http://www.postgresql.org/download/).
2. To configure the connection correctly follow the detailed instructions provided [here](https://eicu-crd.mit.edu/tutorials/install_eicu_locally/ )
3. Open paths.json, adjust the default path locations to locations on your local machine, also do this for eICU_preprocessing/create_all_tables.sql
4. Navigated to the project directory and type in the following commands, keep in mind this step could take a few hours to complete
```shell
    psql 'dbname=eicu user=eicu options=--search_path=eicu'
    
    # Inside the psql console:
    \i eICU_preprocessing/create_all_tables.sql
```

Quit the psql console
```shell
   \q
```

5) The preprocessing will need to be run overnight, it takes approximitely thirteen hours to complete.

```shell
   python3 -m eICU_preprocessing.run_all_preprocessing
```
  
It creates following directory structure:

```bash
eICU_data
├── test
│   ├── diagnoses.csv
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── train
│   ├── diagnoses.csv
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── val
│   ├── diagnoses.csv
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── diagnoses.csv
├── flat_features.csv
├── labels.csv
├── timeseriesaperiodic.csv
├── timeserieslab.csv
├── timeseriesnurse.csv
├── timeseriesperiodic.csv
└── timeseriesresp.csv
```

## Training

1. Preprocessing is complete, all the models can be run in the terminal. Navigate to the directory, TPC-LoS-prediction in the terminal and begin with the following command

```shell
 # This command can be customized using command line arguments
 python3 -m models.run_tpc
```

Running each model will result in the creation of a directory the constains the results and data from the experiment. The directories are named based on the time an experment is ran.

2. Hyperparameter search can be replicated by running the following commands:

```shell
 python3 -m models.hyperparameter_scripts.eICU.tpc
```

Final experiments are located in the directory models/final_experiment_scripts

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

#### Length of Stay Prediction 
- Mean absolute deviation (MAD)
- Mean absolute percentage error (MAPE)
- Mean squared error (MSE)
- Mean squared log error (MSLE)
- Coefficient of determination (R<sup>2</sup>)
- Cohen Kappa Score (Harutyunyan et al. 2019)

For the first four metrics, lower is better. For the last two, higher is better (Rocheteau et al. 2022).

Our model achieves the following performance on :
Below is the result comparison for TPC model between the original paper and our replication experiment on eICU test data.

| **TPC Model**           | **MAD**   | **MSE**   | **MAPE** | **MSLE**  | **R<sup>2</sup>**   | **KAPPA** |
|---------------|-----------|-----------|----------|-----------|-----------|-----------|
| Original Paper | 1.78±0.02 |  21.7±0.5 | 63.5±4.3 | 0.70±0.03 | 0.27±0.02 | 0.58±0.01 |
| Our Results    | 1.71   | 27.76    | 71.63   | 0.74     | 0.24     | 0.62     |

Comparing models to baseline
| **Model**           | **MAD** | **MSE** | **MAPE** | **MSLE** | **R<sup>2</sup>** | **KAPPA** |
| ------------------- | ------- | ------- | -------- | -------- | ----------------- | --------- |
| TPC                 | 2.496   | 24.628  | 197.249  | 0.328    | 0.475             | 0.710     |
| Transformer | 2.62       | 124.74       | 124.74        | 1.44    | 0.13          | 0.35         |
| LSTM  | 2.57        |  30.79       | 119.2         |  1.37        |  0.16         |  0.38        |

Ablations
| **Model**           | **MAD** | **MSE** | **MAPE** | **MSLE** | **R<sup>2</sup>** | **KAPPA** |
| ------------------- | ------- | ------- | -------- | -------- | ----------------- | --------- |
| TPC                 | 2.496   | 24.628  | 197.249  | 0.328    | 0.475             | 0.710     |
| TPC (MSE)         | 2.5       | 24.63       | 197.25        | 1.76        | 0.33  | 0.48         |
| Pointwise 	   | 2.52   | 30.03       | 120.53 | 1.38        | 0.18            | 0.4         |
| Temporal  	| 2.38      |  33.92       | 94.52    |  1.04        |  0.07         |   0.53        |

Original Results
Model | MAD | MAPE | MSE | MSLE | R<sup>2</sup> | Kappa
--- | --- | --- | --- | --- | --- | ---
LSTM | 2.39±0.00 | 118.2±1.1 | 26.9±0.1 | 1.47±0.01 | 0.09±0.00 | 0.28±0.00
CW LSTM | 2.37±0.00 | 114.5±0.4 | 26.6±0.1 | 1.43±0.00 | 0.10±0.00 | 0.30±0.00
Transformer | 2.36±0.00 | 114.1±0.6 | 26.7±0.1 | 1.43±0.00 | 0.09±0.00 | 0.30±0.00
TPC | 1.78±0.02 | 63.5±4.3 | 21.7±0.5 | 0.70±0.03 | 0.27±0.02 | 0.58±0.01

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
