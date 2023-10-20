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
from l2r.common.utils import setup_logging
from loguru import logger
from ruamel.yaml import YAML
from scipy import sparse
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from agents.base import BaseAgent
from agents.models.efficientnet_v2 import EfficientNetV2_FPN_Segmentation
# from agents.utils.localisation_v2 import LocaliseOnTrack
from agents.utils.spatial_mpc import SpatialBicycleModel, SpatialMPC

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


class MrMPC(BaseAgent):
    def __init__(self):
        super(MrMPC).__init__()
        self.start_time = -1
        self.all_speeds = []

        self.cfg = self.load_model_config("models/MPC_agent/params.yaml")
        self.file_logger, self.tb_logger = self.setup_loggers()

        self.birds_eye_view_dimension = 200  # m each length to form a square
        self.bev_scale = 4

        self.MPC_horizon = self.cfg["control"]["horizon"]
        Q = sparse.diags(self.cfg["control"]["step_cost"])  # e_y, e_psi, t
        R = sparse.diags(self.cfg["control"]["r_term"])  # velocity, delta
        QN = sparse.diags(self.cfg["control"]["final_cost"])  # e_y, e_psi, t

        v_min = self.cfg["control"]["speed_profile_constraints"]["v_min"]
        v_max = self.cfg["control"]["speed_profile_constraints"]["v_max"]
        self.wheel_base = self.cfg["vehicle"]["wheel_base"]
        width = self.cfg["vehicle"]["width"]
        delta_max = self.cfg["control"]["input_constraints"]["steering_max"]

        InputConstraints = {
            "umin": np.array([v_min, -np.tan(delta_max) / self.wheel_base]),
            "umax": np.array([v_max, np.tan(delta_max) / self.wheel_base]),
        }
        StateConstraints = {
            "xmin": np.array([-np.inf, -np.inf, -np.inf]),
            "xmax": np.array([np.inf, np.inf, np.inf]),
        }

        model = SpatialBicycleModel(n_states=3, wheel_base=self.wheel_base, width=width)
        self.MPC = SpatialMPC(
            model,
            self.MPC_horizon,
            Q,
            R,
            QN,
            StateConstraints,
            InputConstraints,
            self.cfg["control"]["speed_profile_constraints"],
        )

        seed = self.cfg["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.cfg["evaluation"] or self.cfg["perception"]["model_segmentation"]:
            self.model = self.load_segmentation_model(self.cfg["perception"]["model_path"])

        self.pose = {"velocity": 0}
        self.image_count = 0
        self.enable_collect_images = False

        if self.cfg["data_collection"]["collect_images"] > 0:
            self.enable_collect_images = True
            self.num_image_samples = self.cfg["data_collection"]["collect_images"]
            self.image_save_path = self.save_path + "/datacollection/images"
            self.mask_save_path = self.save_path + "/datacollection/masks"
            self.map_save_path = self.save_path + "/datacollection/maps"
            self.command_save_path = self.save_path + "/datacollection/commands"
            self.command_samples = {}
            paths = [
                self.image_save_path,
                self.mask_save_path,
                self.command_save_path,
                self.map_save_path,
            ]
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)

        self.line_seg_im_w = self.cfg["perception"]["line_seg_im_w"]
        self.line_seg_im_h = self.cfg["perception"]["line_seg_im_h"]

        self.homography = self.get_camera_homography(
            [self.line_seg_im_h, self.line_seg_im_w], self.birds_eye_view_dimension
        )

        self.pose = {"velocity": 0}
        self.previous_steering_command, self._sc, = (
            0,
            0,
        )
        self.previous_acceleration_command, self._ac = 0, 0
        self.maps = {}
        self.current_time = time.time()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.update_control_lock = threading.Lock()
        self.frames_to_display = {}
        self.plotting_thread = threading.Thread(target=self.display, daemon=True)
        self.plotting_thread.start()
        self.thread_exception = None

    @staticmethod
    def load_model_config(path):
        yaml = YAML()
        params = yaml.load(open(path))
        return params

    def setup_loggers(self):
        self.save_path = self.cfg["save_path"] + "_" + self.cfg["experiment_name"]
        save_path = self.save_path + "/logs"
        loggers = setup_logging(save_path, self.cfg["experiment_name"], True)
        loggers[0]("Using random seed: {}".format(0))
        return loggers

    def select_action(self, obs) -> np.array:
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
            action should be in the form of [\delta, a], where \delta is the normalized steering angle,
            and a is the normalized acceleration.
        """
        if self.thread_exception is not None:
            raise self.thread_exception

        if self.start_time == -1:
            self.start_time = time.time()
            self.step_count = 1

        if self.cfg["multithreading"]:
            self.executor.submit(self.maybe_update_control, obs)
        else:
            self.update_control(obs)

        steps_per_second = self.step_count / (time.time() - self.start_time)
        self.step_count += 1
        speed = np.sqrt(obs[0][3] ** 2 + obs[0][4] ** 2 + obs[0][5] ** 2)
        self.all_speeds.append(speed)
        self.average_speed = np.mean(self.all_speeds)
        if self.cfg["debugging"]["verbose"] and self.step_count % 30 == 0:
            # logger.info(f"[RUNTIME INFO] Processing frequency {steps_per_second:.2f}")
            logger.info(f"[RUNTIME INFO] Average speed {self.average_speed:.2f}m/s = {self.average_speed*3.6:.2f}km/h ")

        if self.step_count > 10 and steps_per_second < 2.0:
            if not self.cfg["evaluation"]:
                logger.error(f"[RUNTIME ERROR] Not running fast enough ({steps_per_second:.2f}it/s)")
                raise Exception("Processing too slow, not going to succeed, ABORT")

        if self.step_count > 200 and self.average_speed < 10:
            if not self.cfg["evaluation"]:
                logger.error(
                    f"[RUNTIME ERROR] Car travelling too slow to set decent time ({self.average_speed:.2f}m/s)"
                )
                raise Exception("Vehicle travelling too slow, ABORT")
        _ac = np.clip(self._ac, -1.0, 1.0)
        _sc = np.clip(self._sc, -1.0, 1.0)
        return [_sc, _ac]

    def maybe_update_control(self, obs):
        if not self.update_control_lock.locked():
            with self.update_control_lock:
                try:
                    self.update_control(obs)
                except Exception as e:
                    self.thread_exception = e
        else:
            logger.warning("Threads queuing - skipping observation")

    def update_control(self, obs):
        self.previous_time = self.current_time
        self.current_time = time.time()
        self.previous_steering_command = self._sc
        self.previous_acceleration_command = self._ac
        if isinstance(obs, dict):
            obs = self.preprocess_dict_obs(obs)
        else:
            obs = self.preprocess_observations(obs)

        _sc, _ac = self._step(obs)
        self._sc, self._ac = _sc, _ac

    def preprocess_dict_obs(self, obs):
        input_image = torch.as_tensor(
            copy.deepcopy(obs["CameraFrontRGB"]) / 255,
            dtype=torch.float32,
            device=self.device,
        )
        input_image = input_image.unsqueeze(0).permute(0, 3, 1, 2)
        output = self.model.predict(input_image)
        output = torch.argmax(output, dim=1)
        obs["CameraFrontSegm"] = output.squeeze().cpu().numpy().astype(np.uint8)
        return obs

    def preprocess_observations(self, obs):
        if self.cfg["perception"]["single_cam"]:
            image_names = [
                "CameraFrontRGB",
                "CameraFrontSegm",
            ]
        else:
            image_names = [
                "CameraFrontRGB",
                "CameraLeftRGB",
                "CameraRightRGB",
                "CameraFrontSegm",
                "CameraLeftSegm",
                "CameraRightSegm",
            ]

        if self.cfg["perception"]["model_segmentation"]:
            output_obs = {}

            pose = obs[0]
            processed_pose = self.process_pose(pose)
            output_obs["speed"] = processed_pose["velocity"]
            output_obs["full_pose"] = processed_pose

            images = obs[1]

            for i in range(len(images)):
                output_obs[image_names[i]] = images[i]

            if "CameraFrontSegm" in output_obs and self.cfg["data_collection"]["collect_images"] > 0:
                output_obs["CameraFrontSegm_gt"] = output_obs["CameraFrontSegm"]

            if self.cfg["perception"]["single_cam"]:
                if "CameraLeftRGB" in output_obs.keys():
                    del output_obs["CameraLeftRGB"]
                if "CameraRightRGB" in output_obs.keys():
                    del output_obs["CameraRightRGB"]
                input_image = torch.as_tensor(
                    np.stack(
                        [
                            output_obs["CameraFrontRGB"] / 255,
                        ]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                input_image = input_image.permute(0, 3, 1, 2)
                output = self.model.predict(input_image)
                output = torch.argmax(output, dim=1).cpu().numpy().astype(np.uint8)
                output_obs["CameraFrontSegm"] = output[0]

            else:
                input_image = torch.as_tensor(
                    np.stack(
                        [
                            output_obs["CameraFrontRGB"] / 255,
                            output_obs["CameraLeftRGB"] / 255,
                            output_obs["CameraRightRGB"] / 255,
                        ]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                input_image = input_image.permute(0, 3, 1, 2)
                output = self.model.predict(input_image)
                output = torch.argmax(output, dim=1).cpu().numpy().astype(np.uint8)
                output_obs["CameraFrontSegm"] = output[0]
                output_obs["CameraLeftSegm"] = output[1]
                output_obs["CameraRightSegm"] = output[2]

        else:
            output_obs = {}

            pose = obs[0]
            processed_pose = self.process_pose(pose)
            output_obs["speed"] = processed_pose["velocity"]
            output_obs["full_pose"] = processed_pose

            images = obs[1]

            for i in range(len(images)):
                output_obs[image_names[i]] = images[i]

            for im_key, im in output_obs.items():
                if "Segm" not in im_key:
                    continue
                # output_obs[f"{im_key}_gt"] = copy.deepcopy(output_obs[im_key])
                output_obs[im_key] = np.where(output_obs[im_key] == (109, 80, 204), 1, 0).astype(np.uint8)
                output_obs[im_key] = output_obs[im_key][:, :, 1]

        if output_obs["CameraFrontRGB"].shape[:2] != (
            self.line_seg_im_h,
            self.line_seg_im_w,
        ):
            raise Exception(f"[RUNTIME ERROR] Image shape is {output_obs['CameraFrontRGB'].shape[:2]}")

        return output_obs

    def process_pose(self, pose):
        # Comes in (y, x, z)
        return {
            "SteeringRequest": pose[0],
            "GearRequest": pose[1].astype("float"),
            "Mode": pose[2].astype("float"),
            "velocity": np.sqrt(pose[3] ** 2 + pose[4] ** 2 + pose[5] ** 2),
            "vx": pose[4],
            "vy": pose[3],
            "vz": pose[5],
            "ax": pose[7],
            "ay": pose[6],
            "az": pose[8],
            "avx": pose[10],
            "avy": pose[9],
            "avz": pose[11],
            "yaw": (((np.pi / 2) - pose[12] + np.pi) % (2 * np.pi)) - np.pi,  # Ensure the yaw is within (-pi, pi)
            "pitch": pose[13],
            "roll": pose[14],
            "x": pose[16],
            "y": pose[15],
            "z": pose[17],
        }

    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        Defaults to select_action
        """
        logger.error("[RUNTIME ERROR] RESTART OCCURED (register reset triggered)")
        if self.cfg["data_collection"]["collect_images"] > 0:
            np.save(f"{self.map_save_path}/maps.npy", self.maps)
        return self.select_action(obs)

    def training(self, env):
        """
        Training loop
        - Local development OR Stage 2 'practice' phase
        """
        pass

    def load_model(self, path):
        pass

    def load_segmentation_model(self, path):
        """
        Load model checkpoints.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logger.info(f"[POD INFO] #CPUS: {os.cpu_count()}")
            logger.error("[RUNTIME ERROR] Experiment not running on a gpu")
            raise Exception("MODEL MUST BE ON GPU, ABORT")
        else:
            logger.info(
                f"[POD INFO] #CPUS: {os.cpu_count()} "
                f"Device name: {torch.cuda.get_device_name(0)} "
                f"Max memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB"
            )

        model = EfficientNetV2_FPN_Segmentation(version="efficientnet_v2_s", im_c=3, n_classes=2).to(self.device)
        model.load_state_dict(torch.load(path))
        model.eval()

        return model

    def save_model(self, path):
        """
        Save model checkpoints.
        """
        pass

    @staticmethod
    def get_camera_homography(image_size, bev_dimension):
        camera_information = {
            "CameraFront": {
                "position": [0.000102, 2.099975, 0.7],
                "rotation": [90.0, 0.0, 0.0],
                "calibration_pts": [
                    [-10, 20, 0],
                    [10, 20, 0],
                    [-10, 120, 0],
                    [10, 120, 0],
                ],
            },
            "CameraLeft": {
                "position": [-0.380003, 1.279986, 0.550007],
                "rotation": [110.000002, 0.0, -50.000092],
                "calibration_pts": [[-4, 5, 0], [-4, 20, 0], [-8, 5, 0], [-8, 20, 0]],
            },
            "CameraRight": {
                "position": [0.380033, 1.290036, 0.550005],
                "rotation": [110.000002, 0.0, 50.000092],
                "calibration_pts": [[4, 5, 0], [4, 20, 0], [8, 5, 0], [8, 20, 0]],
            },
        }

        homographies = {}

        for camera, information in camera_information.items():
            height, width = image_size
            camera_matrix = np.float32([[width / 2, 0, width / 2], [0, width / 2, height / 2], [0, 0, 1]])
            rotations = np.flip(information["rotation"])
            rotation_matrix = R.from_euler("zyx", rotations, degrees=True).as_matrix()
            translation_matrix = -np.array(information["position"]).astype(np.float32)

            ground_points = np.array(information["calibration_pts"])

            # World coordinates to camera coordinates
            camera_points = np.add(ground_points, translation_matrix)
            camera_points = np.matmul(rotation_matrix, camera_points.T)

            # Camera coordinates to image coordinates
            camera_points = np.matmul(camera_matrix, camera_points).T
            camera_points = np.divide(camera_points, camera_points[:, 2].reshape(-1, 1))

            ground_points[:, 2] = 1

            homography = cv2.findHomography(camera_points, ground_points)[0]

            # Sanity check
            check = np.matmul(homography, camera_points.T)
            check /= check[2]
            assert np.all(np.isclose(ground_points.T, check)), "Homography calculation is incorrect"

            homographies[camera] = homography

        return homographies

    @staticmethod
    def transform_track_image_points(columns, homography, remove_bottom=511):
        # remove track limit idxs that touch side of image or include bonnet of vehicle
        track_image_coords = np.array(
            [
                [columns[row], row, 1]
                for row in range(len(columns))
                if (columns[row] != 0) and (columns[row] != 511) and row < remove_bottom
            ]
        ).T

        if len(track_image_coords) == 0:
            return np.zeros((2, 0))

        track_ground = np.matmul(homography, track_image_coords)
        track_ground = track_ground[:2] / track_ground[2]

        return track_ground

    def process_image_masks(self, observations, homographies, transform_fn, test=False):
        ascending_array = np.arange(1, observations["CameraFrontSegm"].shape[1] + 1)
        right_track_ground = np.zeros((2, 0))
        left_track_ground = np.zeros((2, 0))
        centre_track_ground = np.zeros((2, 0))

        if "CameraFrontSegm" in observations.keys():
            homography = homographies["CameraFront"]
            seg_mask = observations["CameraFrontSegm"]
            mask = np.multiply(seg_mask, ascending_array)
            right_columns = np.argmax(mask, axis=1)

            mask[mask == 0] = mask.shape[1] + 1
            left_columns = np.argmin(mask, axis=1)

            centre_columns = (right_columns + left_columns) / 2

            if test:
                cv2.imshow("seg_centre", seg_mask * 255)

            # remove bonnet from front camera
            bonnet_row = 288

            right_track_ground = np.append(
                right_track_ground,
                transform_fn(right_columns, homography, bonnet_row),
                axis=1,
            )
            left_track_ground = np.append(
                left_track_ground,
                transform_fn(left_columns, homography, bonnet_row),
                axis=1,
            )

            centre_track_ground = np.append(
                centre_track_ground,
                transform_fn(centre_columns, homography, bonnet_row),
                axis=1,
            )

        if "CameraLeftSegm" in observations.keys():
            homography = homographies["CameraLeft"]
            seg_mask = observations["CameraLeftSegm"]
            mask = np.multiply(seg_mask, ascending_array)
            mask[mask == 0] = mask.shape[1] + 1
            left_columns = np.argmin(mask, axis=1)

            if test:
                cv2.imshow("seg_left", seg_mask * 255)

            left_track_ground = np.append(left_track_ground, transform_fn(left_columns, homography), axis=1)

        if "CameraRightSegm" in observations.keys():
            homography = homographies["CameraRight"]
            seg_mask = observations["CameraRightSegm"]
            mask = np.multiply(seg_mask, ascending_array)
            right_columns = np.argmax(mask, axis=1)

            if test:
                cv2.imshow("seg_right", seg_mask * 255)

            right_track_ground = np.append(right_track_ground, transform_fn(right_columns, homography), axis=1)

        left_track_ground = left_track_ground[:, np.argsort(left_track_ground[1])]
        right_track_ground = right_track_ground[:, np.argsort(right_track_ground[1])]
        centre_track_ground = centre_track_ground[:, np.argsort(centre_track_ground[1])]

        if test:
            cv2.waitKey(0)
            fig, ax = plt.subplots()
            ax.scatter(right_track_ground[0], right_track_ground[1], label="right")
            ax.scatter(left_track_ground[0], left_track_ground[1], label="left")
            ax.scatter(centre_track_ground[0], centre_track_ground[1], label="centre")
            ax.arrow(0, 0, 0, 4, width=0.01)
            ax.legend()
            ax.set_aspect(1)
            plt.show()

        return centre_track_ground, left_track_ground, right_track_ground

    @staticmethod
    def smooth_track_with_polyfit(track, num_points, degree=3):
        ynew = np.linspace(0, np.max(track[1]), num_points)
        coeffs = np.polyfit(track[1], track[0], degree)
        xnew = np.polyval(coeffs, ynew)
        return np.array([xnew, ynew])

    @staticmethod
    def smooth_track_with_spline(track, num_points, smooth_factor=1e4):
        ynew = np.linspace(0, np.max(track[1]), num_points)
        spl = UnivariateSpline(track[1], track[0])
        spl.set_smoothing_factor(smooth_factor)
        xnew = spl(ynew)
        return np.array([xnew, ynew])

    @staticmethod
    def process_track_points(
        centre_track,
        left_track,
        right_track,
        num_points,
        # centre_track,
        interpolate=True,
    ):
        mask = (-50 < left_track[0]) & (left_track[0] < 50) & (0 < left_track[1]) & (left_track[1] < 150)
        left_track = left_track[:, mask]
        mask = (-50 < right_track[0]) & (right_track[0] < 50) & (0 < right_track[1]) & (right_track[1] < 150)
        right_track = right_track[:, mask]
        mask = (-50 < centre_track[0]) & (centre_track[0] < 50) & (0 < centre_track[1]) & (centre_track[1] < 150)
        centre_track = centre_track[:, mask]

        if interpolate:
            # left_track = MPCAgent.interpolate_polyfit_track(left_track, num_points)
            # right_track = MPCAgent.interpolate_polyfit_track(right_track, num_points)
            # centre_track = MrMPC.smooth_track_with_polyfit(centre_track, num_points)
            centre_track = np.concatenate([np.array([[0.0, 0.0]] * 30), centre_track.T]).T
            centre_track = MrMPC.smooth_track_with_spline(centre_track, num_points)

        return (
            centre_track.T,
            left_track.T,
            right_track.T,
        )

    @staticmethod
    def draw_track_lines_on_bev(bev, scale, list_of_lines, colour=(255, 0, 255)):
        for track_line in list_of_lines:
            track_line = (track_line * scale).astype(np.int32)
            track_line[:, 0] = track_line[:, 0] + bev.shape[0] / 2

            for i in range(len(track_line) - 2):
                cv2.line(
                    bev,
                    track_line[i],
                    track_line[i + 1],
                    color=colour,
                    thickness=1,
                )

    @staticmethod
    def transform_track_points(points, translation, rotation):
        points = points - translation
        return np.matmul(rotation, points.T).T

    def save_image(self, name, image):
        if "RGB" in name:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{self.image_save_path}/{name}.png", image)
        else:
            cv2.imwrite(f"{self.mask_save_path}/{name}.png", image)

    def _step(self, obs):
        dt = self.current_time - self.previous_time
        self.pose["velocity"] = obs["speed"]

        centre_track, left_track, right_track = self.process_image_masks(
            obs, self.homography, self.transform_track_image_points
        )
        original_centre_track = copy.copy(centre_track)

        (
            centre_track,
            left_track,
            right_track,
        ) = self.process_track_points(centre_track, left_track, right_track, 200)

        ds = int(len(centre_track) / self.MPC_horizon)
        reference_path = np.stack([centre_track[0::ds, 0], centre_track[0::ds, 1], np.ones(self.MPC_horizon) * 7]).T

        desired_velocity, steering_angle = self.MPC.get_control(reference_path)
        # logger.info(f"[MPC INFO] vel: {desired_velocity:.2f}, st: {steering_angle:.2f}")
        # desired_velocity = self.MPC.accelerations[0]
        desired_velocity = np.mean(self.MPC.projected_control[0, 0])
        _ac = np.clip((desired_velocity - self.pose["velocity"]) / 4, -16, 6)
        steering_angle = np.mean(self.MPC.projected_control[1, 0])
        _sc = np.clip(steering_angle / 0.3, -1, 1)

        if self.cfg["debugging"]["show_control_image"]:
            bev_size = self.birds_eye_view_dimension * self.bev_scale
            bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

            self.draw_track_lines_on_bev(bev, 4, [centre_track], colour=(0, 255, 0))
            self.draw_track_lines_on_bev(bev, 4, [original_centre_track.T], colour=(255, 0, 255))
            x = self.MPC.current_prediction[0]
            y = self.MPC.current_prediction[1]
            self.draw_track_lines_on_bev(bev, 4, [np.stack([x, y], axis=0).T], colour=(0, 0, 255))

            bev = cv2.flip(bev, 0)
            self.frames_to_display["MPC_controller"] = bev

        if self.cfg["debugging"]["show_camera"] and not self.cfg["test"]:
            image = cv2.cvtColor(obs["CameraFrontRGB"], cv2.COLOR_BGR2RGB)
            self.frames_to_display["CameraFrontRGB"] = image

        if self.cfg["debugging"]["show_segmentation"] and not self.cfg["test"]:
            self.frames_to_display["CameraFrontSegm"] = obs["CameraFrontSegm"] * 255

        # Save results or collect data
        if self.enable_collect_images:
            filename = self.image_count

            for key, image in obs.items():
                if "RGB" in key or "Segm" in key:
                    self.save_image(key + f"_{self.image_count}", image)

            self.command_samples[filename] = [
                dt,
                _sc,
                _ac,
                self.pose["velocity"],
                obs["full_pose"],
            ]
            current_map = {
                "centre": centre_track,
                "left": left_track,
                "right": right_track,
            }

            np.save(f"{self.map_save_path}/{filename}.npy", current_map)

            with open(f"{self.command_save_path}/commands.json", "w+") as command_file:
                command_file.write(json.dumps(self.command_samples))

            if self.image_count == self.num_image_samples - 1:
                self.enable_collect_images = False
                logger.info("Image collection completed")

            self.image_count += 1

        return [_sc, _ac]

    # def display(self):
    #     while True:
    #         for window_name in self.frames_to_display.keys():
    #             cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #             cv2.imshow(window_name, self.frames_to_display[window_name])
    #             cv2.waitKey(1)
    #         time.sleep(0.5)
