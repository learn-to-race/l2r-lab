"""Store schema from older config system. DO NOT USE."""
from strictyaml import (
    load,
    Map,
    Str,
    Int,
    Seq,
    Enum,
    Float,
    Optional,
    Bool,
    CommaSeparated,
)

# Config YAML Schema
cv_trainer_schema = Map(
    {
        "cv_trainer_config": Map(
            {
                "batch_size": Int(),
                "num_epochs": Int(),
                "lr": Float(),
                "model_save_path": Str(),
                "train_data_path": Str(),
                "val_data_path": Str(),
            }
        )
    }
)
