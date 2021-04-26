# Scotland-Stop-And-Search-Prediction

This project aims to develop a machine learning classifier to predict stop and search outcomes in scotland. It uses published data from Police on individual stop and search instances combined with the Scottish Index of Multiple Deprivation to provide information on the area the stop and search took place in.

## Police Scotland Stop and Search

This project was developed with the April - December 2020 stop and search dataset, published on the [Police Scotland website](https://www.scotland.police.uk/about-us/police-scotland/stop-and-search/data-publication/)

## SIMD

Scottish Index Multiple Deprivation v2 2020

[SIMD](http//simd.scot])

[Technical Notes](https://www.gov.scot/binaries/content/documents/govscot/publications/statistics/2020/09/simd-2020-technical-notes/documents/simd-2020-technical-notes/simd-2020-technical-notes/govscot%3Adocument/SIMD%2B2020%2Btechnical%2Bnotes.pdf)

# Downloading Data

Within the "Get Data" directory, there is a "get_data.py" python file that will download the data sets used.

It will also scrape some additional data on data zones and electoral wards from [statistics.gov.scot](https://statistics.gov.scot/home) that are needed for aggregating the SIMD data in the larger electoral ward zones.

# Data Cleaning

Two Jupyter notebooks in [2 Cleaning](2 Cleaning)
