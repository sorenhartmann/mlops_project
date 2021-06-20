
import pytest
from src.data.substitutions import apply_substitutions
import pandas as pd

@pytest.mark.parametrize("input_str,output_str", [
    ("a .. b", "a ... b"),
    ("whatever http://t.co, fdf", "whatever URL, fdf"),
])
def test_substitutions(input_str, output_str):

    result = apply_substitutions(pd.Series([input_str]))

    assert type(result) is pd.Series
    assert result[0] == output_str

