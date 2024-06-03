# DSL_project

## Quick Usage

### requirement

In order to training the model and run the code, you need to install some libraries and packages in advance.

Run this command to install everything
```commandline
pip install -r requirements.txt
```

### pre-analysis
In our report, we did some analysis before introducing our methods. There are three figures in this section in the report. 
Run ```pre_analysis.py``` to reproduce these three figures. Results will be saved in folder ```Figures```. 
Figures we used in the report are already in this folder, but you can still run again to generate by yourself.

The data ```Data/processed_peptides10.csv``` and ```Data/prot_sequences_df.csv``` are already in ```Data``` folder. 
All blosum scores between every two peptides in the second dataset ```blosum_no_repeat.npy``` need to be calculated locally, because the file is too huge.
After calculation , it will also be saved in ```Data``` folder.

### search pipeline
You can just run ```predict_pipeline.py``` to get the final result. The final result will be saved in ```Result/mmp9_search.csv``` and ```Result/mmp9_search.csv```. 

Our results are already in ```Result``` folder, but you can still run again.

mmp3 result samples

```commandline
,top1,top5,top10,target scores,margins,maximum other scores
YYYVAQIM,17.0,15.8,14.6,2.52,1.01,1.51
APASVRNA,17.0,16.8,16.3,2.23,0.8,1.43
QRKIRIKG,19.0,16.6,15.4,2.03,0.6399999999999999,1.39
RPQAVVKM,19.0,16.4,15.5,2.11,0.69,1.42
ARSVRLGG,19.0,17.8,17.1,1.66,0.7599999999999999,0.9
KPVAMTKM,20.0,19.2,18.1,1.66,0.4399999999999999,1.22
RLAPLASS,20.0,18.4,17.2,2.54,1.05,1.49
FAISMVVK,20.0,17.4,15.9,2.13,0.8199999999999998,1.31
RPVAYANR,21.0,19.4,18.0,2.18,1.4400000000000002,0.74
...
```

mmp9 result samples

```commandline
,top1,top5,top10,target scores,margins,maximum other scores
VVRALSAR,16.0,15.0,14.3,1.88,0.4399999999999999,1.44
IVRAIIAS,17.0,15.6,14.7,2.12,0.51,1.61
MTRMHIAS,17.0,16.2,15.1,1.84,0.7100000000000002,1.13
TLRSLIRS,17.0,17.0,15.8,2.01,0.5499999999999998,1.46
VVRALAGQ,18.0,17.4,16.3,1.75,1.1,0.65
MLRFLWDM,18.0,15.8,14.6,1.77,0.5,1.27
QLSFLQQI,18.0,16.4,15.6,1.81,0.77,1.04
WAKMLKNI,18.0,17.0,15.9,2.16,0.55,1.61
NFLHLLML,19.0,17.0,15.6,1.88,0.6699999999999999,1.21
...
```

### predict pipeline
#### models training
#### use models trained by us
To save the time of training the models again, the result from step 1 is already saved in ```Result/pred_mmp3.npy``` and ```Result/pred_mmp3.npy```, for mmp9 and mmp3 respectively.
You can just run ```predict_pipeline.py``` to get the final result. The final result will be saved in ```Result/mmp9_predict.csv``` and ```Result/mmp9_predict.csv```. 

Our results are already in ```Result``` folder, but you can still run again.

mmp3 results samples:

```commandline
,top1,top5,top10
SLQACKLA,15.0,14.2,13.3
RAGTVRRQ,16.0,14.2,13.4
FKGNQFWA,16.0,14.8,13.9
QHAPYFKG,16.0,15.2,14.5
SLGPQVAE,16.0,15.4,15.2
LLAFVCDI,16.0,15.6,14.9
VLGHSERR,16.0,15.8,15.1
SPGAVLRA,16.0,15.8,15.3
NPNACRAS,16.0,16.0,15.1
SVAYVSRS,16.0,16.0,15.7
...
```

mmp9 results samples:

```commandline
,top1,top5,top10
VLRELRCV,15.0,14.6,14.3
SCRHLQFI,16.0,14.6,13.6
FHRKYRIP,16.0,15.4,14.7
NSYTIKGL,16.0,15.8,15.1
LGFLQRSS,17.0,16.6,16.0
VAATPTSL,19.0,16.4,15.5
LSQVEVIL,19.0,17.4,15.5
GLGVSAGA,19.0,17.4,16.5
VATELRCQ,19.0,18.4,16.9
...
```


## Files Overview

Our project has many files, some of which are python files and some of which are jupyter files. Overall, the python files are mainly for multiple experiments and contain the code used for the experiments, while the jupter notebook is mainly used to analyse and visualise the results of the experiments. Next we will briefly describe the role of each file.

### model_training.py

This file focuses on training the FC (2 layers) model with the full training set. The model is named BioNN in our code framework. 

### extract_features.py

This file is used to extract all data features

### dl_experiment.py

Validate the Deep Learning Models by KFold Cross-Validation

### Result_Analysis.ipynb

Analysis of the validation result

### ml_experiment.ipynb

This file tests the performance of several machine learning models on the MMP dataset. The models include linear regression, decision tree regression, XGBoost, CatBoost, and a simple neural network. We use 5-fold cross-validation to evaluate the performance of the models. The evaluation metrics include mean squared error, mean absolute error, root mean squared error, area under the curve, f1 score, precision, and recall. The results are saved in the `Result` folder.

### extract_unique_peptides.ipynb

This file is based on the model trained in the model_training.py file, with peptides unique to mmp3 and mmp9 extracted from the merops dataset.

 ### process_merops_data.ipynb

This file is used to process the Merops Data.

### models.py

This file store the structure of different DL models, including FC (2 layers), FC(4 layers) and ResNet

### feature_experiment.ipynb

This file use Ridge regression model to evaluate the importance of different features.