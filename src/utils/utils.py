from dataclasses import dataclass
import os, sys
import logging, re
import numpy as np
from datetime import datetime


def find_envvar_patterns(self, config, key):
    pattern = re.compile(".*?${(\w+)}.*?")
    try:
        envvars = re.findall(pattern, config[key])
    except:
        envvars = []
        pass
    return envvars


def replace_envvar_patterns(self, config, key, envvars, args):
    for i, var in enumerate(envvars):
        if var == "DIRHASH":
            dirhash = "{}/".format(args.dirhash) if not args.runtime == "local" else ""
            config[key] = config[key].replace("${" + var + "}", dirhash)
        if var == "PREFIX":
            prefix = {"local": "/data", "phoebe": "/mnt"}
            config[key] = config[key].replace("${" + var + "}", prefix[args.runtime])
        else:
            config[key] = config[key].replace(
                "${" + var + "}", os.environ.get(var, var)
            )


def resolve_envvars(self, config, args):

    for key in list(config.keys()):

        if isinstance(config[key], dict):
            # second level
            for sub_key in list(config[key].keys()):
                sub_envvars = find_envvar_patterns(config[key], sub_key)
                if len(sub_envvars) > 0:
                    for sub_var in sub_envvars:
                        replace_envvar_patterns(config[key], sub_key, sub_envvars, args)

        envvars = find_envvar_patterns(config, key)
        if len(envvars) > 0:
            replace_envvar_patterns(config, key, envvars, args)

    return config


def is_number(self, s):
    """
    Somehow, the most pythonic way to check string for float number; used for safe user input parsing
    src: https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


class RecordExperience:
    def __init__(self, record_dir, track, experiment_name, logger, agent=False):

        self.record_dir = record_dir
        self.track = track
        self.experiment_name = experiment_name
        self.filename = "transition"
        self.agent = agent
        self.logger = logger

        self.path = os.path.join(self.record_dir, self.track, self.experiment_name)

        self.logger("Recording agent experience")

    def save(self, record):

        filename = f"{self.path}/{record['stage']}/{record['episode']}/{self.filename}_{self.experiment_name}_{record['step']}"

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(
            os.path.join(self.path, record["stage"], str(record["episode"])),
            exist_ok=True,
        )

        np.savez_compressed(filename, **record)

        return record

    def save_thread(self):
        """Meant to be run as a separate thread"""
        if not self.agent:
            raise Exception("RecordExperience requires an SACAgent")

        while True:
            batch = self.agent.save_queue.get()
            self.logger("[RecordExperience] Saving experience.")
            for record in batch:
                self.save(record)


@dataclass
class ActionSample:
    """Generic object to store action-related params. Might be useful to remove."""

    action = None
    value = None
    logp = None
