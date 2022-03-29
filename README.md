# TDT4310 - Project
Repository for the course project for `TDT4310 - Intelligent Text Analytics and Language Understanding`.

The dataset that will be used here is found on kaggle: https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews.

## Setup

### Environment setup

I recommend using a `conda` environment for this repository. The requirements are found in `requirements.txt` and can be used in conda by running:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --name <env> --file requirements.txt
```

### Other setup

The data is not included in this repository, due to limitations in git. The project however assumes that the data is put in a seperate folder called `data`.
