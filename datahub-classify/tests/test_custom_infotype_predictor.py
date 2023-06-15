import random
import string
import uuid
from datetime import datetime, timedelta

import pytest

from datahub_classify.helper_classes import ColumnInfo, Metadata
from datahub_classify.infotype_predictor import predict_infotypes
from datahub_classify.reference_input import input1 as default_config


def random_vehicle_number():
    state_codes = ["MH", "TN", "BH", "DL"]
    separators = ["-", " ", "", "_"]
    return "".join(
        [
            random.choice(state_codes),
            random.choice(separators),
            str(random.randint(1, 20)),
            random.choice(separators),
            "".join(random.choices(string.ascii_letters, k=random.randint(1, 3))),
            random.choice(separators),
            "".join(random.choices(string.digits, k=4)),
        ]
    )


@pytest.fixture(scope="module")
def column_infos():
    return [
        ColumnInfo(
            metadata=Metadata(
                meta_info={
                    "Name": "id",
                    "Description": "Primary",
                    "Datatype": "str",
                    "Dataset_Name": "entry_register",
                }
            ),
            values=[uuid.uuid4() for i in range(1, 100)],
        ),
        ColumnInfo(
            metadata=Metadata(
                meta_info={
                    "Name": "vehicle_number",
                    "Description": "Vehicle registration number ",
                    "Datatype": "str",
                    "Dataset_Name": "entry_register",
                }
            ),
            values=[random_vehicle_number() for i in range(1, 100)],
        ),
        ColumnInfo(
            metadata=Metadata(
                meta_info={
                    "Name": "entry_time",
                    "Description": "Time of vehicle's entry",
                    "Datatype": "datetime",
                    "Dataset_Name": "entry_register",
                }
            ),
            values=[
                datetime.now() - timedelta(hours=random.randint(0, 24))
                for i in range(1, 100)
            ],
        ),
    ]


@pytest.fixture
def custom_config_patch():
    return {
        "IN_Vehicle_Registration_Number": {
            "Prediction_Factors_and_Weights": {
                "Name": 0.2,
                "Description": 0.1,
                "Datatype": 0.1,
                "Values": 0.6,
            },
            "Name": {
                "regex": [
                    "^.*vehicle.*num.*$",
                    "^.*license.*plat.*num.*$",
                    "^.*license.*plat.*num.*$",
                    "^.*vehicle.*plat.*num.*$",
                    "^.*vehicle.*num.*plat.*$",
                ]
            },
            "Description": {
                "regex": [
                    "^.*vehicle.*num.*$",
                    "^.*license.*plat.*num.*$",
                    "^.*license.*plat.*num.*$",
                    "^.*vehicle.*plat.*num.*$",
                    "^.*vehicle.*num.*plat.*$",
                ]
            },
            "Datatype": {"type": ["str", "varchar", "text"]},
            "Values": {
                "prediction_type": "regex",
                "regex": [r"[a-z]{2}[-_\s]?[0-9]{1,2}[-_\s]?[a-z]{2,3}[-_\s]?[0-9]{4}"],
                "library": [],
            },
        }
    }


def test_custom_infotype_prediction(column_infos, custom_config_patch):
    # Default config
    out_column_infos = predict_infotypes(
        column_infos, confidence_level_threshold=0.7, global_config=default_config
    )
    assert not out_column_infos[0].infotype_proposals
    assert not out_column_infos[1].infotype_proposals
    assert not out_column_infos[2].infotype_proposals

    # Config with new custom infotype, all factors
    config_new = default_config.copy()
    config_new.update(custom_config_patch)
    out_column_infos = predict_infotypes(
        column_infos, confidence_level_threshold=0.7, global_config=config_new
    )
    assert not out_column_infos[0].infotype_proposals
    assert not out_column_infos[2].infotype_proposals

    predicted_infotypes = out_column_infos[1].infotype_proposals
    assert predicted_infotypes
    assert len(predicted_infotypes) == 1
    assert predicted_infotypes[0].infotype == "IN_Vehicle_Registration_Number"
    assert predicted_infotypes[0].debug_info.name == 1
    assert predicted_infotypes[0].debug_info.description == 1
    assert predicted_infotypes[0].debug_info.datatype == 1
