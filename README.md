# Behaviour Classification

This repository is the projectory for classification task for the SportsTracker project. This project aims to classifying human behaviour recognition while human do golf activity.

Now, it can classify
- walking
- stand alone
- golf swing
using the ANN and CNN structure.

It didn't tested yet in the large size of dataset.

***

### Data Requirement

In the data folder, it should contain the experimental data for running the preprocessing.m

- `(subject)_trial(#)_label.txt` files
    - In current data folder, (subject)_trial(#).m files save hand-labeling data of the experiment as txt form.
- `(subjct)/traial(#).csv` files

***

### Running

When all data requirements are satisfied
1. run prprocessing.m
    - creates test.txt, train.txt, test_label.txt, train_label.txt
2. run train.py
    - Learning neural net model using train dataset
    - creates ANN.pt, CNN.pt
3. run eval.py
    - Evaluate classification model using test dataset
