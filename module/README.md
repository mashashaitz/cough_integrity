#### Cough Validation ML-algorithm

A **cough integrity classification** algorithm, that rejects cough-recordings with either:
* two little cough, or
* loud background noise.

It consists of a Pipeline with tree major steps:
  1. Principal Component Analysis, 
  2. Scaler,
  3. Support Vector Classification GridSearch with custom Cross Validation, based on patient distribution.

The train-test set distribution is 80% to 20% respectively, featuring more than 5000 recordings from 6 different sources. The classification is based on embeddings of said recordings. The best ROC AUC attained on test-set is 0,97. 

###### MainConfig.py and supporting_functions.py files containing data structure information and most links are missing for NDA-related concerns.
