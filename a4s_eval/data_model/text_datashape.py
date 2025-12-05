import uuid
import pytest
import pandas as pd
from a4s_eval.data_model.evaluation import DataShape, Feature, FeatureType


@pytest.fixture
def text_datashape():
    """
    Test suite expects this fixture but does not define it.
    We create a minimal DataShape with two text features:
    text_original and text_transformed.
    """
    return DataShape(
        features=[
            Feature(
                pid=uuid.uuid4(),
                name="text_original",
                feature_type=FeatureType.TEXT,
                min_value=None,
                max_value=None,
            ),
            Feature(
                pid=uuid.uuid4(),
                name="text_transformed",
                feature_type=FeatureType.TEXT,
                min_value=None,
                max_value=None,
            )
        ],
        target=None,
        date=None
    )
