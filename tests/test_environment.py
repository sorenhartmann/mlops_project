import sys
import pandas as pd
import re

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


def test_data():
    train_df = pd.read_csv('data/preprocessed/train.csv')
    test_df = pd.read_csv('data/preprocessed/test.csv')

    try:
        assert len(train_df) == 7613
        assert len(test_df) == 3263
    except AssertionError:
        print('Full data not loaded')

def test_sub():

    test_str = "aren't isn't What's haven't hasn't There's He's It's You're"

    test_str = re.sub(r"aren't", "are not", test_str)
    test_str = re.sub(r"isn't", "is not", test_str)
    test_str = re.sub(r"What's", "What is", test_str)
    test_str = re.sub(r"haven't", "have not", test_str)
    test_str = re.sub(r"hasn't", "has not", test_str)
    test_str = re.sub(r"There's", "There is", test_str)
    test_str = re.sub(r"He's", "He is", test_str)
    test_str = re.sub(r"It's", "It is", test_str)
    test_str = re.sub(r"You're", "You are", test_str)

    try:
        assert test_str == "are not is not What is have not has not There is He is It is You are"
    except AssertionError:
        print('Substitution test not correct.')






if __name__ == '__main__':
    main()
    test_data()
    test_sub()
