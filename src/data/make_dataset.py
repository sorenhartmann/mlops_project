# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from src.data.data_cleaning import clean


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():  #input_filepath, output_filepath
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    #Retrieve raw data from kaggle:
    api = KaggleApi()
    api.authenticate()
    api.competition_download_file('nlp-getting-started',
                                  file_name='train.csv',
                                  path='data/raw/')
    api.competition_download_file('nlp-getting-started',
                                  file_name='test.csv',
                                  path='data/raw/')

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #Read raw data:
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')

    #Fill no location and keyword with "no_location", "no_keyword"
    for df in [train, test]:
        for col in ['keyword', 'location']:
            df[col] = df[col].fillna(f'no_{col}')

    #Clean text using data_cleaning.py
    train['text_cleaned'] = train['text'].apply(lambda s: clean(s))
    test['text_cleaned'] = test['text'].apply(lambda s: clean(s))

    #Dump new processed data:
    train.to_csv('data/preprocessed/train.csv')
    test.to_csv('data/preprocessed/test.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
