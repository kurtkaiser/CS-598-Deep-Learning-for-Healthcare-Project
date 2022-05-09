# Project Information

The aim of this project is to understand, replicate and extend the paper ['Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit'](https://arxiv.org/pdf/2007.09483v4.pdf) by Emma Rocheteau, Pietro Li, and
Stephanie Hyland ( [repository](https://github.com/EmmaRocheteau/TPC-LoS-prediction)). The paper focuses on creating a deep learning model that will aid in hospital bed management, a daily challenge for the healtcare industry. 

This repository contains a modified version of the orginal work, done in order to carry out and extend the experiments.
## Requirements
To reproduct this experment, it is necessarily to install Python, to meet some of the requirements listed above, it is necessary to utilize version 3.7 or above. Python can be downloaded and installed directly from [here](https://www.python.org/). 

### Specification of Dependencies
In order to preform our experiements, particular versions of various libraries and tools must be used. The following is the required versions, this information is contained within the file requirements.txt. We conducted our experments on a Windows based and a Linux based machine.

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

### Data Download
In order to reproduce this the experements the eICU data must be downloaded from the eICU Collaborative Reserach Database (version 2.0). Credentials, earned through completing an online course, are required to access. When credentialized, the data can be access through [Physionet.org]( https://physionet.org/content/eicu-crd/2.0/).

### Setup eICU Database Locally
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

1. Preprocessing is complete, all the models can be run in the terminal. Navigate to the directory, TPC-LoS-prediction in the terminal and begin with the following command. The models other than tpc that can be run are listed in the models directory.

```shell
 # This command can be customized using command line arguments. See initisalise_arguments.py for all parsable arguments.
 python3 -m models.run_tpc
```

Running each model will result in the creation of a directory the constains the results and data from the experiment. The directories are named based on the time an experment is ran.

2. Hyperparameter search can be replicated by running the following commands:

```shell
 python3 -m models.hyperparameter_scripts.eICU.tpc
```

Final experiments are located in the directory models/final_experiment_scripts

## Evaluation

To evaluate the trained [tpc] model, run:

```eval
python3 -m models.run_tpc --mode test
```
CLAIM 1
The results of the paper and ours were in congruence. For the main temporal pointwise convolution (TPC) model, the author’s mean average deviation (MAD) in days of stay were 1.78, while ours was 1.71, Although 1.71 is outside their margin of error of 0.02, it’s congruent with their claim that TPC is performing better than the MAD of best-performing baseline models listed (Transformer) by 18 - 68%. As seen in below tables, the other metrics studied, MSE, MAPE, MSLE, R^2, and KAPPA also reproduced similar results in relative performance.

Claim 2
Using MSLE instead of MSE as the loss function did deliver better performance, with a MAD of 2.42 (2.21 in authors’ results) using MSE vs. the 1.71 (1.78) listed above. Other than MSE (as expected), the performance was also similar or superior with MSLE as the loss function.

## Results

#### Length of Stay Prediction 
- Mean absolute deviation (MAD)
- Mean absolute percentage error (MAPE)
- Mean squared error (MSE)
- Mean squared log error (MSLE)
- Coefficient of determination (R<sup>2</sup>)
- Cohen Kappa Score (Harutyunyan et al. 2019)

For the first four metrics, lower is better. For the last two, higher is better (Rocheteau et al. 2022).

Below is the result comparison for TPC model between the original paper and our replication experiment on eICU test data.

| **TPC Model**           | **MAD**   | **MSE**   | **MAPE** | **MSLE**  | **R<sup>2</sup>**   | **KAPPA** |
|---------------|-----------|-----------|----------|-----------|-----------|-----------|
| Original Paper | 1.78±0.02 |  21.7±0.5 | 63.5±4.3 | 0.70±0.03 | 0.27±0.02 | 0.58±0.01 |
| Our Results    | 1.71   | 27.76    | 71.63   | 0.74     | 0.24     | 0.62     |

Comparing models to baseline
| **Model**           | **MAD** | **MSE** | **MAPE** | **MSLE** | **R<sup>2</sup>** | **KAPPA** |
| ------------------- | ------- | ------- | -------- | -------- | ----------------- | --------- |
| TPC                 | 1.71   | 27.76    | 71.63   | 0.74     | 0.24     | 0.62     |
| Transformer | 2.62       | 124.74       | 124.74        | 1.44    | 0.13          | 0.35         |
| LSTM  | 2.57        |  30.79       | 119.2         |  1.37        |  0.16         |  0.38        |

Ablations
| **Model**           | **MAD** | **MSE** | **MAPE** | **MSLE** | **R<sup>2</sup>** | **KAPPA** |
| ------------------- | ------- | ------- | -------- | -------- | ----------------- | --------- |
| TPC                | 1.71   | 27.76    | 71.63   | 0.74     | 0.24     | 0.62     |
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
