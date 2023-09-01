"""New auto-config system, to make easy configuration management."""
try:
    from typing import get_type_hints, TypedDict, get_origin, get_args
except ImportError:
    from typing_extensions import get_type_hints, TypedDict, get_origin, get_args
import inspect
import strictyaml as sl
import yaml
from enum import Enum
import importlib


def yamlize(configurable_class):
    """Class decorator for configuration. See any of the configurable classes for more.

    Args:
        configurable_class (type): Some class to add configuration parameters to.
    """

    def convert_type_to_strictyaml(val):
        """Convert generic typing type to strictyaml validator

        Args:
            val (type): Type parameter from docstring

        Raises:
            ValueError: When type is not currently convertible.

        Returns:
            sl.validator: strictyaml validator for YAML checking.
        """

        type_container = get_origin(val)
        if type_container == tuple:
            args = get_args(val)
            arg_list = [convert_type_to_strictyaml(arg) for arg in args]
            return sl.FixedSeq(arg_list)
        elif type_container == list:
            args = get_args(val)
            arg_list = [convert_type_to_strictyaml(arg) for arg in args]
            return sl.Seq(arg_list[0])
        elif type_container is not None:
            raise ValueError(
                f"Type origin {type_container} could not be converted to StrictYAML type. Please add to the convert_type_to_strictyaml function."
            )

        if val == int:
            return sl.Int()
        elif val == str:
            return sl.Str()
        elif val == float:
            return sl.Float()
        elif val == bool:
            return sl.Bool()
        elif val == ConfigurableDict:
            return sl.Map({"name": sl.Str(), "config": sl.Any()})
        else:
            raise ValueError(
                f"Type {val} could not be converted to StrictYAML type. Please add to the convert_type_to_strictyaml function."
            )

    init_types = get_type_hints(configurable_class.__init__)
    init_signature = inspect.signature(configurable_class.__init__)
    init_defaults = {
        k: v.default
        for k, v in init_signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    schema = {}
    for key, val in init_types.items():
        if key == "return":
            continue
        if key in init_defaults:
            key = sl.Optional(key, default=init_defaults[key])
        schema[key] = convert_type_to_strictyaml(val)
    schema = sl.Map(schema)
    configurable_class.schema = schema

    def init_from_config_dict(cls, config):
        """Initialize class from config dictionary

        Args:
            config (dictionary): Create instance of class from dictionary.

        Returns:
            cls: Object from class and config.
        """
        config_yamlized = yaml.dump(config)
        config_dict = sl.load(config_yamlized, schema).data
        return cls(**config_dict)

    configurable_class.instantiate_from_config_dict = classmethod(init_from_config_dict)

    def init_from_config(cls, config_file_location):
        """Initialize class from config file

        Args:
            config_file_location (path): Path to config file

        Raises:
            ValueError: Error loading file

        Returns:
            cls: object from class.
        """
        with open(config_file_location, "r") as mf:
            yaml_str = mf.read()
        try:
            config_dict = sl.load(yaml_str, schema).data
        except Exception as e:
            raise ValueError(yaml_str, schema, e)
        try:
            return cls(**config_dict)
        except Exception as e:
            import logging

            logging.info(config_dict)
            raise ValueError(config_dict) from e

    configurable_class.instantiate_from_config = classmethod(init_from_config)
    print(configurable_class)
    return configurable_class


class NameToSourcePath(Enum):
    """Map Enum from name to source path location.

    Args:
        Enum (str): Path to import class from.
    """

    buffer = "src.buffers"
    encoder = "src.encoders"
    logger = "src.loggers"
    runner = "src.runners"
    agent = "src.agents"
    network = "src.networks"
    planner = "src.agents.petsplanners"


class ConfigurableDict(TypedDict):
    """Dict specification for nested config files."""

    name: str
    config: dict


def create_configurable(config_yaml, name_to_path):
    """Create configurable object from config file path and source location

    Args:
        config_yaml (str): Config yaml location
        name_to_path (NameToSourcePath): Map from class type to source location

    Returns:
        object: Instantiated object.
    """
    name_to_path = str(name_to_path.value)
    schema = sl.Map({"name": sl.Str(), "config": sl.Any()})
    with open(config_yaml, "r") as mf:
        yaml_contents = mf.read()
        config_dict = sl.load(yaml_contents, schema).data
    cls = getattr(importlib.import_module(name_to_path), config_dict["name"])

    return cls.instantiate_from_config_dict(config_dict["config"])


def create_configurable_from_dict(config_dict, name_to_path):
    """Create configurable object from config dict and soruce location

    Args:
        config_dict (dict): Config dict
        name_to_path (NameToSourcePath): Map from class type to source location

    Returns:
        object: Instantiated object.
    """
    name_to_path = str(name_to_path.value)
    schema = sl.Map({"name": sl.Str(), "config": sl.Any()})
    yaml_contents = yaml.dump(config_dict)
    config_dict = sl.load(yaml_contents, schema).data
    cls = getattr(importlib.import_module(name_to_path), config_dict["name"])
    return cls.instantiate_from_config_dict(config_dict["config"])
