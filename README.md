# Paper Reproduction Study - TPC Networks for LoS
Format reference: https://github.com/paperswithcode/releasing-research-code

The aim of this project is to understand, replicate and extend the paper the paper ['Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit'](https://arxiv.org/pdf/2007.09483v4.pdf) by Emma Rocheteau, Pietro Li, and
Stephanie Hyland ( [repository](https://github.com/EmmaRocheteau/TPC-LoS-prediction)). The paper focuses on creating a deep learning model that will aid in hospital bed management, a daily challenge for the healtcare industry. 

This repository contains a modified version of the orginal work, done in order to carry out and extend the experiments.

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

To reproduct this experment, it is necessarily to install Python, to meet some of the requirements listed above, it is necessary to utilize version 3.7 or above. Python can be downloaded and installed directly from  [here](https://www.python.org/). 

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


#### Download Data

Download eICU Data from Physionet.org: https://physionet.org/content/eicu-crd/2.0/. 


### Citations

```bibtex
@inproceedings{rocheteau2021,
author = {Rocheteau, Emma and Li\`{o}, Pietro and Hyland, Stephanie},
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