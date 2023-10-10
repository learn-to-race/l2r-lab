import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize


class MPCAgent(BaseAgent):
    default_action_space = gym.spaces.Box(-1, 1, (2,))

    def __init__(self, action_space=default_action_space):
        self.action_space = action_space

    def select_action(self, obs) -> np.array:  # pragma: no cover
        """
        # Outputs action given the current observation
        obs: a dictionary
            During local development, the participants may specify their desired observations.
            During evaluation on AICrowd, the participants will have access to
            obs =
            {
              'CameraFrontRGB': front_img, # numpy array of shape (width, height, 3)
              'CameraLeftRGB': left_img, # numpy array of shape (width, height, 3)
              'CameraRightRGB': right_img, # numpy array of shape (width, height, 3)
              'track_id': track_id, # integer value associated with a specific racetrack
              'speed': speed # float value of vehicle speed in m/s
            }
        returns:
            action: np.array (2,)
            action should be in the form of [\delta, a], where \delta is the normalized steering angle, and a is the normalized acceleration.
        """
        raise NotImplementedError

    def register_reset(self, obs) -> np.array:  # pragma: no cover
        """
        Same input/output as select_action, except this method is called at episodal reset.
        Defaults to select_action
        """
        return self.select_action(obs)

    def update(self, data):  # pragma: no cover
        """
        Model update, given data
        """
        raise NotImplementedError

    def load_model(self, path):  # pragma: no cover
        """
        Load model checkpoints.
        """
        raise NotImplementedError

    def load_segmentation_model():
        ## Load the Segmentation model here check with Kevin
        # model = EfficientNetV2_FPN_Segmentation(version="efficientnet_v2_s", im_c=3, n_classes=2).to(self.device)
        # model.load_state_dict(torch.load(path))
        # model.eval()

        # return model
        raise NotImplementedError

    def save_model(self, path):  # pragma: no cover
        """
        Save model checkpoints.
        """
        raise NotImplementedError
