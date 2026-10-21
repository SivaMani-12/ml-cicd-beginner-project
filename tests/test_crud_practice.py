from src.data import make_dataset, validate


def test_dataset_has_target_column():
    """The dataset must contain the target column used for training."""
    df = make_dataset()

    assert "churned" in df.columns


def test_dataset_has_expected_features():
    """The dataset must contain all expected feature columns."""
    df = make_dataset()

    expected_columns = {
        "tenure_months",
        "monthly_charges",
        "support_tickets",
        "contract_type",
        "uses_auto_pay",
        "churned",
    }

    assert set(df.columns) == expected_columns


def test_dataset_passes_validation():
    """A generated dataset should pass the data validation checks."""
    df = make_dataset()

    validate(df)


def test_dataset_is_not_empty():
    """The generated dataset should contain records."""
    df = make_dataset()

    assert not df.empty


def test_dataset_has_no_missing_values():
    """The generated dataset should not contain missing values."""
    df = make_dataset()

    assert not df.isnull().any().any()


def test_dataset_target_contains_only_binary_values():
    """The target must contain only 0 and 1."""
    df = make_dataset()

    assert set(df["churned"].unique()).issubset({0, 1})


def test_dataset_is_reproducible():
    """Using the same random seed should generate identical data."""
    df1 = make_dataset(seed=42)
    df2 = make_dataset(seed=42)

    assert df1.equals(df2)


def test_dataset_validation_rejects_missing_values():
    """Validation should reject data containing missing values."""
    df = make_dataset()
    df.loc[0, "monthly_charges"] = float("nan")

    try:
        validate(df)
    except ValueError as exc:
        assert "missing values" in str(exc).lower()
    else:
        raise AssertionError("Expected validation to reject missing values")
