import pytest
from src.data import make_dataset,validate
def test_data_shape_and_validity():
    d=make_dataset(100); assert len(d)==100; validate(d)
def test_data_is_reproducible():
    assert make_dataset().equals(make_dataset())
def test_missing_values_rejected():
    d=make_dataset(30); d.loc[0,"monthly_charges"]=float("nan")
    with pytest.raises(ValueError,match="missing"): validate(d)
def test_invalid_target_rejected():
    d=make_dataset(30); d.loc[0,"churned"]=8
    with pytest.raises(ValueError,match="only 0 and 1"): validate(d)
