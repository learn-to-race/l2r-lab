import math
import time

import matplotlib.pyplot as plt
import numpy as np
import osqp
from matplotlib import markers
from scipy import sparse


class SpatialBicycleModel:
    def __init__(self, n_states, wheel_base, width):
        self.n_states = n_states
        self.length = wheel_base
        self.width = width
        self.safety_margin = width / 2

    def t2s(self, reference_waypoint, reference_state):
        """
        Convert spatial state to temporal state. Either convert self.spatial_
        state with current waypoint as reference or provide reference waypoint
        and reference_state.
        :return Spatial State equivalent to reference state
        """
        ref_x, ref_y, ref_psi = (
            reference_waypoint["x"],
            reference_waypoint["y"],
            reference_waypoint["psi"],
        )
        x, y, psi = reference_state
        # Compute spatial state variables
        if isinstance(reference_state, np.ndarray):
            e_y = np.cos(ref_psi) * (y - ref_y) - np.sin(ref_psi) * (x - ref_x)
            e_psi = psi - ref_psi

            # Ensure e_psi is kept within range (-pi, pi]
            e_psi = np.mod(e_psi + math.pi, 2 * math.pi) - math.pi
        else:
            print("Reference State type not supported!")
            e_y, e_psi = None, None
            exit(1)

        # time state can be set to zero since it's only relevant for the MPC
        # prediction horizon
        t = 0.0

        return [e_y, e_psi, t]

    def s2t(self, reference_waypoint, reference_state):
        """
        Convert spatial state to temporal state given a reference waypoint.
        :param reference_waypoint: waypoint object to use as reference
        :param reference_state: state vector as np.array to use as reference
        :return Temporal State equivalent to reference state
        """

        # Compute temporal state variables
        if isinstance(reference_state, np.ndarray):
            x = reference_waypoint["x"] - reference_state[0] * np.sin(
                reference_waypoint["psi"]
            )
            y = reference_waypoint["y"] + reference_state[0] * np.cos(
                reference_waypoint["psi"]
            )
            psi = reference_waypoint["psi"] + reference_state[1]
        else:
            print("Reference State type not supported!")
            x, y, psi = None, None, None
            exit(1)

        return [x, y, psi]

    def linearize(self, v_ref, kappa_ref, delta_s):
        """
        Linearize the system equations around provided reference values.
        :param v_ref: velocity reference around which to linearize
        :param kappa_ref: kappa of waypoint around which to linearize
        :param delta_s: distance between current waypoint and next waypoint
        """

        ###################
        # System Matrices #
        ###################

        # Construct Jacobian Matrix
        a_1 = np.array([1, delta_s, 0])
        a_2 = np.array([-(kappa_ref**2) * delta_s, 1, 0])
        a_3 = np.array([-kappa_ref / v_ref * delta_s, 0, 1])

        b_1 = np.array([0, 0])
        b_2 = np.array([0, delta_s])
        b_3 = np.array([-1 / (v_ref**2) * delta_s, 0])

        f = np.array([0.0, 0.0, 1 / v_ref * delta_s])

        A = np.stack((a_1, a_2, a_3), axis=0)
        B = np.stack((b_1, b_2, b_3), axis=0)

        return f, A, B


class SpatialMPC:
    def __init__(
        self,
        model,
        N,
        Q,
        R,
        QN,
        StateConstraints,
        InputConstraints,
        SpeedProfileConstraints,
    ):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param N: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param ay_max: maximum allowed lateral acceleration in curves
        """

        # Parameters
        self.N = N  # horizon
        self.Q = Q  # weight matrix state vector
        self.R = R  # weight matrix input vector
        self.QN = QN  # weight matrix terminal

        # Model
        self.model = model

        # Dimensions
        self.nx = self.model.n_states
        self.nu = 2

        # Precision
        self.eps = 1e-12

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        self.SpeedProfileConstraints = SpeedProfileConstraints

        # Maximum lateral acceleration
        self.ay_max = self.SpeedProfileConstraints["ay_max"]

        # Current control and prediction
        self.current_prediction = None

        # Counter for old control signals in case of infeasible problem
        self.infeasibility_counter = 0

        # Current control signals
        self.current_control = np.zeros((self.nu * self.N))

        # Initialize Optimization Problem
        self.optimizer = osqp.OSQP()

    def compute_speed_profile(self, reference_path, Constraints):
        """
        Compute a speed profile for the path. Assign a reference velocity
        to each waypoint based on its curvature.
        :param Constraints: constraints on acceleration and velocity
        curvature of the path
        """

        # Set optimization horizon
        N = len(reference_path)

        # Constraints
        a_min = np.ones(N - 1) * Constraints["a_min"]
        a_max = np.ones(N - 1) * Constraints["a_max"]
        v_min = np.ones(N) * Constraints["v_min"]
        v_max = np.ones(N) * Constraints["v_max"]

        # Maximum lateral acceleration
        ay_max = Constraints["ay_max"]

        # Inequality Matrix
        D1 = np.zeros((N - 1, N))

        # Iterate over horizon
        for i in range(N):
            # Get information about current waypoint
            current_waypoint = reference_path[i]
            # distance between waypoints
            li = current_waypoint["dist_ahead"]
            # curvature of waypoint
            ki = current_waypoint["kappa"]

            # Fill operator matrix
            # dynamics of acceleration
            if i < N - 1:
                D1[i, i : i + 2] = np.array([-1 / (2 * li), 1 / (2 * li)])

            # Compute dynamic constraint on velocity
            v_max_dyn = np.sqrt(ay_max / (np.abs(ki) + self.eps))
            if v_max_dyn < v_max[i]:
                v_max[i] = v_max_dyn

        # v_max[-1] = min(10.0, v_max[-1])

        # Construct inequality matrix
        D1 = sparse.csc_matrix(D1)
        D2 = sparse.eye(N)
        D = sparse.vstack([D1, D2], format="csc")

        # Get upper and lower bound vectors for inequality constraints
        l = np.hstack([a_min, v_min])
        u = np.hstack([a_max, v_max])

        # Set cost matrices
        P = sparse.eye(N, format="csc")
        q = -1 * v_max

        # Solve optimization problem
        problem = osqp.OSQP()
        problem.setup(P=P, q=q, A=D, l=l, u=u, verbose=False)
        speed_profile = problem.solve().x

        # Assign reference velocity to every waypoint
        for i, wp in enumerate(reference_path):
            wp["v_ref"] = speed_profile[i]

        self.speed_profile = speed_profile
        return reference_path

    def construct_waypoints(self, waypoint_coordinates):
        """
        Reformulate conventional waypoints (x, y) coordinates into waypoint
        objects containing (x, y, psi, kappa, ub, lb)
        :param waypoint_coordinates: list of (x, y) coordinates of waypoints in
        global coordinates
        :return: list of waypoint objects for entire reference path
        """

        # List containing waypoint objects
        waypoints = []

        # Iterate over all waypoints
        for wp_id in range(len(waypoint_coordinates) - 1):
            # Get start and goal waypoints
            current_wp = np.array(waypoint_coordinates[wp_id])[:-1]
            next_wp = np.array(waypoint_coordinates[wp_id + 1])[:-1]
            width = np.array(waypoint_coordinates[wp_id + 1])[-1]

            # Difference vector
            dif_ahead = next_wp - current_wp

            # Angle ahead
            psi = np.arctan2(dif_ahead[1], dif_ahead[0])

            # Distance to next waypoint
            dist_ahead = np.sqrt(dif_ahead[0] ** 2 + dif_ahead[1] ** 2)

            # Get x and y coordinates of current waypoint
            x, y = current_wp[0], current_wp[1]

            # Compute local curvature at waypoint
            # first waypoint

            prev_wp = np.array(waypoint_coordinates[wp_id - 1][:-1])
            dif_behind = current_wp - prev_wp
            angle_behind = np.arctan2(dif_behind[1], dif_behind[0])
            angle_dif = np.mod(psi - angle_behind + math.pi, 2 * math.pi) - math.pi
            kappa = angle_dif / (dist_ahead + self.eps)

            if wp_id == 0:
                kappa = 0
            elif wp_id == 1:
                waypoints[0]["kappa"] = kappa

            waypoints.append(
                {
                    "x": x,
                    "y": y,
                    "psi": psi,
                    "kappa": kappa,
                    "dist_ahead": dist_ahead,
                    "width": width,
                }
            )

        return waypoints

    def update_prediction(self, spatial_state_prediction, reference_path):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(self.N):
            # Get associated waypoint
            associated_waypoint = reference_path[n]
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(
                associated_waypoint, spatial_state_prediction[n, :]
            )

            # Save predicted coordinates in world coordinate frame
            x_pred.append(predicted_temporal_state[0])
            y_pred.append(predicted_temporal_state[1])

        return x_pred, y_pred

    def _init_problem(self, spatial_state, reference_path):
        """
        Initialize optimization problem for current time step.
        """
        self.N = len(reference_path)

        # Constraints
        umin = self.input_constraints["umin"]
        umax = self.input_constraints["umax"]
        xmin = self.state_constraints["xmin"]
        xmax = self.state_constraints["xmax"]

        # LTV System Matrices
        A = np.zeros((self.nx * (self.N + 1), self.nx * (self.N + 1)))
        B = np.zeros((self.nx * (self.N + 1), self.nu * (self.N)))
        # Reference vector for state and input variables
        ur = np.zeros(self.nu * self.N)
        xr = np.zeros(self.nx * (self.N + 1))
        # Offset for equality constraint (due to B * (u - ur))
        uq = np.zeros(self.N * self.nx)
        # Dynamic state constraints
        xmin_dyn = np.kron(np.ones(self.N + 1), xmin)
        xmax_dyn = np.kron(np.ones(self.N + 1), xmax)
        # Dynamic input constraints
        umax_dyn = np.kron(np.ones(self.N), umax)
        # umax_dyn[:2] = 0
        # Get curvature predictions from previous control signals
        kappa_pred = (
            np.tan(np.array(self.current_control[3::] + self.current_control[-1:]))
            / self.model.length
        )

        # Iterate over horizon
        for n in range(self.N):
            # Get information about current waypoint
            current_waypoint = reference_path[n]
            delta_s = current_waypoint["dist_ahead"]
            kappa_ref = current_waypoint["kappa"]
            v_ref = current_waypoint["v_ref"]

            # Compute LTV matrices
            f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s)
            A[
                (n + 1) * self.nx : (n + 2) * self.nx, n * self.nx : (n + 1) * self.nx
            ] = A_lin
            B[
                (n + 1) * self.nx : (n + 2) * self.nx, n * self.nu : (n + 1) * self.nu
            ] = B_lin

            # Set reference for input signal
            ur[n * self.nu : (n + 1) * self.nu] = np.array([v_ref, kappa_ref])
            # Compute equality constraint offset (B*ur)
            uq[n * self.nx : (n + 1) * self.nx] = (
                B_lin.dot(np.array([v_ref, kappa_ref])) - f
            )

            # Constrain maximum speed based on predicted car curvature
            vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n]) + 1e-12))
            if vmax_dyn < umax_dyn[self.nu * n]:
                umax_dyn[self.nu * n] = vmax_dyn

        ub = (
            np.array([reference_path[i]["width"] / 2 for i in range(self.N)])
            - self.model.safety_margin
        )
        lb = (
            np.array([-reference_path[i]["width"] / 2 for i in range(self.N)])
            + self.model.safety_margin
        )
        xmin_dyn[0] = spatial_state[0]
        xmax_dyn[0] = spatial_state[0]
        xmin_dyn[self.nx :: self.nx] = lb
        xmax_dyn[self.nx :: self.nx] = ub

        # Set reference for state as center-line of drivable area
        xr[self.nx :: self.nx] = (lb + ub) / 2

        # Get equality matrix
        Ax = sparse.kron(
            sparse.eye(self.N + 1), -sparse.eye(self.nx)
        ) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        # Get inequality matrix
        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        # Combine constraint matrices
        A = sparse.vstack([Aeq, Aineq], format="csc")

        # Get upper and lower bound vectors for equality constraints
        lineq = np.hstack([xmin_dyn, np.kron(np.ones(self.N), umin)])
        uineq = np.hstack([xmax_dyn, umax_dyn])
        # Get upper and lower bound vectors for inequality constraints
        x0 = np.array(spatial_state)
        leq = np.hstack([-x0, uq])
        ueq = leq
        # Combine upper and lower bound vectors
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Set cost matrices
        P = sparse.block_diag(
            [
                sparse.kron(sparse.eye(self.N), self.Q),
                self.QN,
                sparse.kron(sparse.eye(self.N), self.R),
            ],
            format="csc",
        )
        q = np.hstack(
            [
                -np.tile(np.diag(self.Q.A), self.N) * xr[: -self.nx],
                -self.QN.dot(xr[-self.nx :]),
                -np.tile(np.diag(self.R.A), self.N) * ur,
            ]
        )

        # Initialize optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self, reference_path, offset=0):
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """
        self.reference_path = self.construct_waypoints(reference_path)
        self.reference_path = self.compute_speed_profile(
            self.reference_path, self.SpeedProfileConstraints
        )

        # Number of state variables
        nx = self.model.n_states
        nu = 2

        # x, y psi (y axis is forward)
        state = np.array([offset, 0, np.pi / 2])

        # Update spatial state
        spatial_state = self.model.t2s(
            reference_state=state, reference_waypoint=self.reference_path[0]
        )

        # Initialize optimization problem
        self._init_problem(spatial_state, self.reference_path)

        # Solve optimization problem
        dec = self.optimizer.solve()

        if dec.info.status == "solved":
            # Get control signals
            control_signals = np.array(dec.x[-self.N * nu :])
            control_signals[1::2] = np.arctan(control_signals[1::2] * self.model.length)
            v = control_signals[2]
            delta = control_signals[3]

            # Update control signals
            all_velocities = control_signals[0::2]
            all_delta = control_signals[1::2]
            self.projected_control = np.array([all_velocities, all_delta])

            # self.current_control = control_signals

            # Get predicted spatial states
            x = np.reshape(dec.x[: (self.N + 1) * nx], (self.N + 1, nx))

            # Update predicted temporal states
            self.current_prediction = self.update_prediction(x, self.reference_path)

            self.times = np.diff(x[:, 2])

            self.accelerations = np.diff(x[:, 0]) / np.diff(x[:, 2])
            self.steer_rates = np.diff(x[:, 1]) / np.diff(x[:, 2])

            # Get current control signal
            u = np.array([v, delta])

            # if problem solved, reset infeasibility counter
            self.infeasibility_counter = 0

        else:
            print("Infeasible problem. Previously predicted" " control signal used!")
            id = nu * (self.infeasibility_counter + 1)
            u = np.array(self.current_control[id : id + 2])

            # increase infeasibility counter
            self.infeasibility_counter += 1

        if self.infeasibility_counter == (self.N - 1):
            print("No control signal computed!")
            exit(1)

        return u


def rotate_track_points(x, y, angle):
    path_rotation = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(path_rotation, np.stack([x, y]))


def get_hairpin_track(radius, number_of_points, angle=0):
    x = np.cos(np.linspace(0, 3 / 2 * np.pi, number_of_points)) * radius - radius
    y = np.sin(np.linspace(0, 3 / 2 * np.pi, number_of_points)) * radius

    x, y = rotate_track_points(x, y, angle)

    return x, y


def get_curved_track(coeff, number_of_points, angle=0):
    x = np.linspace(0, 100, number_of_points)
    y = coeff * x**2

    x, y = rotate_track_points(x, y, angle)

    return y, x


def get_chicane_track(distance_to_chicane, chicane_width, number_of_points, angle=0):
    y = np.linspace(0, 100, number_of_points)
    x = chicane_width / (1 + np.exp(-0.1 * (y - distance_to_chicane)))

    x, y = rotate_track_points(x, y, angle)

    return x, y


def get_straight_track(length, number_of_points, angle=0):
    x = np.zeros(number_of_points)
    y = np.linspace(0, length, number_of_points)

    x, y = rotate_track_points(x, y, angle)

    return x, y


def main():
    N = 50
    Q = sparse.diags([0.001, 0.0, 0.0])  # e_y, e_psi, t
    R = sparse.diags([1e-6, 10])  # velocity, delta
    QN = sparse.diags([0.001, 0.0, 0.01])  # e_y, e_psi, t

    road_width = 1000
    path_type = "hairpin"  # "hairpin", "curve", "chicane"

    v_max = 60.0  # m/s
    wheel_base = 2.898
    width = 2.5
    delta_max = 0.3  # rad original 0.66
    ay_max = 2  # m/s^2
    a_min = -0.1  # m/s^2
    a_max = 0.1  # m/s^2

    InputConstraints = {
        "umin": np.array([0.0, -np.tan(delta_max) / wheel_base]),
        "umax": np.array([v_max, np.tan(delta_max) / wheel_base]),
    }
    StateConstraints = {
        "xmin": np.array([-np.inf, -np.inf, 0]),
        "xmax": np.array([np.inf, np.inf, np.inf]),
    }

    SpeedProfileConstraints = {
        "a_min": a_min,
        "a_max": a_max,
        "v_min": 0.0,
        "v_max": v_max,
        "ay_max": ay_max,
    }

    model = SpatialBicycleModel(n_states=3, wheel_base=wheel_base, width=width)
    controller = SpatialMPC(
        model, N, Q, R, QN, StateConstraints, InputConstraints, SpeedProfileConstraints
    )

    # Experiment settings
    colours = ["b", "c", "k", "g", "m", "y", "r"]
    show_example_by_example = False
    angle = 0.1
    experiments = 7

    # Curve experiments
    quadratic_coeff = 0.02
    curve_coefficient = np.linspace(-quadratic_coeff, quadratic_coeff, experiments)

    # Constant radius experiments
    test_radii = np.linspace(10, 100, experiments)

    # Chicane experiments
    chicane_width = 40
    distance_to_chicane = np.linspace(40, 100, experiments)

    # Straight line experiments
    line_of_sight = np.linspace(40, 200, experiments)

    for path_type in ["hairpin", "chicane", "curve", "straight"]:
        # for path_type in ["curve"]:
        fig, ax = plt.subplots(1)
        fig1, ax1 = plt.subplots(2)
        ax = [ax]

        for i in range(experiments):
            controller = SpatialMPC(
                model,
                N,
                Q,
                R,
                QN,
                StateConstraints,
                InputConstraints,
                SpeedProfileConstraints,
            )

            if path_type == "hairpin":
                x, y = get_hairpin_track(test_radii[i], N, -np.pi / 6)
            elif path_type == "chicane":
                x, y = get_chicane_track(
                    distance_to_chicane[i], chicane_width, N, angle
                )
            elif path_type == "curve":
                x, y = get_curved_track(curve_coefficient[i], N, angle)
            elif path_type == "straight":
                x, y = get_straight_track(line_of_sight[i], N, angle)

            test_reference_path = np.stack(
                [
                    x,
                    y,
                    np.ones(N) * road_width,
                ]
            ).T

            st = time.time()
            control_output = controller.get_control(test_reference_path, offset=0.0)
            print(f"Time to solve get_control: {time.time() - st:.4f}")

            # print(controller.current_prediction)
            cum_dist = np.cumsum(
                [controller.reference_path[i]["dist_ahead"] for i in range(N - 1)]
            )
            print(controller.current_control)
            print(control_output)

            if show_example_by_example:
                ax[0].clear()
                ax1[0].clear()
                ax1[1].clear()

            ax[0].scatter(
                test_reference_path[:, 0],
                test_reference_path[:, 1],
                c="g",
                label="reference",
            )

            ax[0].scatter(
                controller.current_prediction[0],
                controller.current_prediction[1],
                c=colours[i],
                label="predicted",
            )
            cum_dist = np.concatenate([[0], cum_dist])
            ax1[0].set_title("Velocity reference: green, Planned velocity: red, m/s")
            ax1[0].plot(controller.projected_control[0], c=colours[i])
            ax1[0].plot(controller.speed_profile, "--", c=colours[i])

            ax1[1].set_title("Steering command (rad)")
            ax1[1].plot(controller.projected_control[1], c=colours[i])

            ax[0].set_aspect(1)

            if show_example_by_example:
                plt.pause(0.5)

        if not show_example_by_example:
            plt.show()
            plt.close()


if __name__ == "__main__":
    main()
