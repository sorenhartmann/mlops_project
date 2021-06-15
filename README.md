Machine Learning Operations Project
==============================

Project Description
------------
The overall goal of this project is to use transformers for predicting whether a given tweet is about a real disaster or not. We intend to ensure reproducibility of the code using the principles presented in the MLOps course. The experiment visualizer `wandb` will be used for creating dashboards showing the main results of this project. Additionally, profilers may be used for identifying possible bottlenecks in the code.

The framework we will be using is [Transformers](https://github.com/huggingface/transformers) from the Pytorch Ecosystem. A pre-trained model provided by the repository will be used to perform NLP tasks such as feature extraction and prediction. This includes [ConvBERT](https://arxiv.org/abs/1910.01108) from the transformers framework which is an autoencoding model that has achived good performance in natural language processing (NLP) tasks while reducing model parameters and training time compared to the [BERT](https://arxiv.org/abs/1810.04805) model. 

Data is from the competition "Natural Language Processing with Disaster Tweets" found on [Kaggle](https://www.kaggle.com/c/nlp-getting-started/data). It consists of the text from a tweet, a keyword from that tweet and the location where that tweet was sent from. It should be noted that the keyword and location may be blank. These data set features will be used to classify which tweets are about real disasters and which ones are not. The data set contains 7,613 and 3,263 observations for training and testing respectively.  

If time permits, we will submit our contribution to the Kaggle competition. 


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
