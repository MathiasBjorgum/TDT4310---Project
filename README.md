# TDT4310 - Project
Repository for the course project for `TDT4310 - Intelligent Text Analytics and Language Understanding`.

The dataset that will be used here is found on kaggle: https://www.kaggle.com/datasets/aaron7sun/stocknews.

## Setup

This project requieres some libraries listed in...

The data is not included in this repository, due to limitations in git. The project however assumes that the data is put in a seperate folder called `data`.

In theory this project should work with other datasets than the one described here. The important thing is that the data is in the correct format. I will here list how the different datasets should be structured:

### Stock data

The stock datas should be on a format retrieved from https://finance.yahoo.com/. The most important part is that the `.csv` file includes the headers `Open` and `Close`.
