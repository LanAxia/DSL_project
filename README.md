# DSL_project

## Project Abstract



## Quick Usage

### pre-analysis
In our report, we did some analysis before introducing our methods. There are three figures in this section in the report. 
Run ```pre_analysis.py``` to reproduce these three figures. Results will be saved in folder ```Figures```.

The data ```Data/processed_peptides10.csv``` and ```Data/prot_sequences_df.csv``` are already in ```Data``` folder. 
All blosum scores between every two peptides in the second dataset ```blosum_no_repeat.npy``` need to be calculated locally, because the file is too huge.
After calculation , it will also be saved in ```Data``` folder.

### search pipeline

### predict pipeline

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