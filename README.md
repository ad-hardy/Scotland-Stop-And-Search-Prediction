# Scotland-Stop-And-Search-Prediction

This project aims to develop a machine learning classifier to predict stop and search outcomes in scotland. It uses published data from Police on individual stop and search instances combined with the Scottish Index of Multiple Deprivation to provide information on the area the stop and search took place in.

## Data

### Police Scotland Stop and Search

This project was developed with the April - December 2020 stop and search dataset, published on the [Police Scotland website](https://www.scotland.police.uk/about-us/police-scotland/stop-and-search/data-publication/)

### SIMD

Scottish Index Multiple Deprivation v2 2020

[SIMD](http//simd.scot])

[Technical Notes](https://www.gov.scot/binaries/content/documents/govscot/publications/statistics/2020/09/simd-2020-technical-notes/documents/simd-2020-technical-notes/simd-2020-technical-notes/govscot%3Adocument/SIMD%2B2020%2Btechnical%2Bnotes.pdf)

## Downloading Data

Within the [Get Data](https://github.com/adhardy/Scotland-Stop-And-Search-Prediction/tree/main/1%20Get%20Data) directory, there is a "get_data.py" python file that will download the data sets used.

It will also scrape some additional data on data zones and electoral wards from [statistics.gov.scot](https://statistics.gov.scot/home) that are needed for aggregating the SIMD data in the larger electoral ward zones.

## Data Cleaning

Two Jupyter notebooks in [Cleaning](https://github.com/adhardy/Scotland-Stop-And-Search-Prediction/tree/main/2%20Cleaning) walk through the data cleaning and exploratory data analysis process and explain the decisions made.

This directory also contains eda.py which is a set of functions to help with the cleaning and EDA process.

## Logistic Regression

The most thoroughly explored classifier is an SKLearn logistic regressor in [logistic_regression.ipynb](https://github.com/adhardy/Scotland-Stop-And-Search-Prediction/blob/main/3%20Logistic%20Regression/logistic_regression.ipynb). This notebook goes through a forward selection of features from the two datasets, evaluates key metrics from the selected model, optimises the threshold choice and compares against the test dataset.

## Decision Trees

A short exploration a decision tree model is contained in [decision_tree.ipynb](https://github.com/adhardy/Scotland-Stop-And-Search-Prediction/blob/main/4%20Other%20Models/decision_tree.ipynb).

# Key Findings

Logistic Regression produced the model with the highest performance with an ROC AUC score of 0.63 on the test data. It struggled to make correct predictions with a ture positive rate of 0.68 and true negative rate of 0.50.

The main problem was a lack of signal amongst the available features, with only a handful having any predictive power:

- Age, gender and ethnicity had only the smallest amount of predictive ower (a suprising result in itself).
- The deprivation factors in an area from the deprivation index (simd.scot; includes education, crime rate, housing...) also have no predictive power.
- The biggest predictors are where you were stopped (which police subdivision) and the reason you were stopped (with suspicion of drugs and warrants being the biggest predictors).
