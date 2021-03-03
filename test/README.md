#### A notebook with two major cough-validation classifier tests:

##### TEST 1:
The comparison of manual labelling vs classifier labelling of 42 randomly chosen previously unlabelled files from datasets previously used to train the classifier. 
Manual labelling is done with approximate quality gradation, where: 
* 0 stands for not cough, 
* 0.25 stands for cough with loud background noise, 
* 0.5 stands for cough with slight background noise, 
* 0.75  stands for little cough/quiet cough, 
* 1 stands for clear cough that lasts longer than 5 seconds,
* Every result equal to or above 0.5 is deemed acceptable for the classifier.

We have attained:
  1. Standard deviation equal to 0.13,
  2. ROC AUC equal to 0.92.

##### TEST 2:
The comparison of manual labelling vs classifier labelling of 105 random chosen previously unlabelled files from an independent test-set.
Manual labelling is done with approximate quality gradation, where: 
* 0 stands for not cough, 
* 0.25 stands for cough with loud background noise, 
* 0.5 stands for cough with slight background noise, 
* 0.75  stands for little cough/quiet cough, 
* 1 stands for clear cough that lasts longer than 5 seconds,
* Every result equal to or above 0.5 is deemed acceptable for the classifier.

We have attained:
  1. Standard deviation equal to 0.098,
  2. P-value equal to 0.000000000001,
  3. ROC AUC equal to 0.899.

###### All data and paths cannot be featured for NDA-related reasons.
