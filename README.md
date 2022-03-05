abalone_age_classification
==============================

The goal of this project is to predict the age of abalones from physical measurements. 

The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age (https://archive.ics.uci.edu/ml/datasets/abalone)

By classifying the age of abalones from measurements, we remove the need for the costly task of counting rings. The age in years is set as the number of rings plus 1.5. I first approached this project as a regression model, as can be found in the notebooks folder. Since the models were not performing well, I decided to attack the problem as a classification case, considering age as a category with the classes young, middle age, and old. The cuts for the age category were decided after analyzing the distribution of the original variable.


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── predicted      <- Predicted values.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── mlruns             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── config         <- Scripts to set project parameters
    │   │   └── config.yml
    │   │   └── core.py
    │   │
    │   ├── train.py       <- Script to train model
    │   │
    │   ├── predict.py     <- Script to train make predictions from saved model
    │   │
    │   └── utils.py       <- Script with utility functions
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
