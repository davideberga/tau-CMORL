from collections import deque
import sys

from .env_utils import check_path_collision
from matplotlib import patches
import numpy as np
import torch
from gym import spaces
import rtamt
from matplotlib import pyplot as plt
from .dynamics_cmorl import DynamicsSimulator
from enum import Enum


plt.rcParams.update({"font.size": 12})
device = "cuda" if torch.cuda.is_available() else "cpu"


class STATE_MODE(Enum):
    NPC = 1
    RELATIVE = 2
    RELATIVE_STLGYM = 3
    MPC = 4
    RELAXATION = 5


class ENV_MODE(Enum):
    NORMAL = 1
    HARD = 2
    ONLY_WALLS = 3


ENV_MODE_HUMAN = {1: "normal", 2: "hard", 3: "only_walls"}


class RoverEnvCMORLSafety:
    def __init__(
        self,
        args,
        mode: STATE_MODE,
        env_mode: ENV_MODE = ENV_MODE.NORMAL,
        train: bool = False,
        max_steps: int = 200,
        time_horizon: int = None,
    ):
        self.action_space = spaces.Box(
            np.array([0, -np.pi], dtype=np.float32),
            np.array([1.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        n_subformulas = 3
        self.observation_space = spaces.Box(
            np.array(
                [0 for _ in range(7)]
                + [-4, 0, -4, 0, -7]
                + [-0.5 for _ in range(n_subformulas)]
            ),
            np.array(
                [1 for _ in range(7)]
                + [4, 1, 4, 1, 5]
                + [0.5 for _ in range(n_subformulas)]
            ),
            dtype=np.float32,
        )

        self.beta = 100.0
        self.pure_state_length = 12
        self.tau = max_steps if time_horizon is None else time_horizon

        self.past_tau_trajectory = {}
        self.episode_trajectory = {}
        self.past_raw_trajectory = deque(maxlen=self.tau)
        self.smoothing_factor = 500.0

        self.max_steps = max_steps
        self.args = args
        self.num_steps = 0  # step counter
        self.sample_idx = 0
        self.render_mode = "human"
        self.mode = mode
        self.train = train
        self.env_mode = env_mode

        self.complete_state = np.array([])
        self.history = []
        self.state = np.array([])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sim = None
        self.init_battery = -1

        self.close_enough_charger = 0.08
        self.close_enough_target = 0.08
        self.safe_distance = 0.2
        self.minimum_battery = 2
        self.rules = self.build_stl()

    def init_simulator(self):
        beam_angles = torch.tensor(
            [
                -torch.pi / 2,
                -torch.pi / 3,
                -torch.pi / 4,
                0.0,
                torch.pi / 4,
                torch.pi / 3,
                torch.pi / 2,
            ]
        ).to(device)
        self.sim: DynamicsSimulator = DynamicsSimulator(
            wait_for_charging=4,
            area_h=10,
            area_w=10,
            squared_area=True,
            beam_angles=beam_angles,
            device=self.device,
            close_thres=self.close_enough_charger,
        )
        self.chargers = self.sim.static_chargers()
        if self.env_mode == ENV_MODE.NORMAL:
            _, self.obs, _, _ = self.sim.generate_objects()
            # if self.train:
            #     self.init_battery = 2
        if self.env_mode == ENV_MODE.HARD:
            _, self.obs, _, _ = self.sim.generate_objects_hard()
            self.chargers = self.sim.static_chargers_hard()
            self.init_battery = 5
        if self.env_mode == ENV_MODE.ONLY_WALLS:
            _, self.obs, _, _ = self.sim.generate_objects_walls_only()
            self.init_battery = 1

        self.init_states = self.sim.initialize_x(
            1, self.obs, self.chargers, test=not self.train
        )
        self.obstacles = self.init_states[1][1:]

    def fix_state_for(self, state, robot_pose, target, charger):
        torch_state = torch.cat((robot_pose, target, charger))
        numpy_state = torch_state.cpu().numpy()

        self.complete_state = np.concatenate((state, numpy_state))
        self.pose = robot_pose

        # Save history for path visualization
        npc_state = np.concatenate(
            (numpy_state[0:2], numpy_state[3:5], numpy_state[5:7], state[-2:])
        )
        self.history.append(npc_state)

        if not self.past_tau_trajectory["battery"]:
            # Fill with invalid states
            # print(state)
            for _ in range(self.tau - 1):
                self.append_to_traj_dict(
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 1, 0, 1, state[11]]
                )
            self.append_to_traj_dict(state)
        return self.preprocess(self.past_tau_trajectory, state)

    def reset(
        self,
    ):
        if self.sim is None or self.sample_idx > 5000 - 3:
            self.init_simulator()

        self.reset_trajectories()

        state, _, robot_pose, target, charger = self.sim.initialize_x(
            1, self.obs, self.chargers, test=not self.train
        )
        state = state.squeeze()

        if self.init_battery != -1:
            state[-2] = self.init_battery

        self.target = target.squeeze()
        self.chargers = charger.unsqueeze(0)

        state = state.cpu().numpy()
        self.state = self.fix_state_for(
            state, robot_pose.squeeze(), self.target, charger.squeeze()
        )

        # Reset history and set intial state
        torch_state = torch.cat((robot_pose.squeeze(), self.target, charger.squeeze()))
        numpy_state = torch_state.cpu().numpy()
        npc_state = np.concatenate(
            (numpy_state[0:2], numpy_state[3:5], numpy_state[5:7], state[-2:])
        )
        self.history = [npc_state]

        self.sample_idx += 1
        self.num_steps = 0

        # self.render()
        return self.state

    def reset_trajectories(self):
        to_observe = [
            "battery",
            "target_distance",
            "charger_distance",
            "lidar_min",
        ]
        self.past_tau_trajectory["time"] = range(self.tau)
        self.episode_trajectory["time"] = []
        for var in to_observe:
            self.past_tau_trajectory[var] = deque(maxlen=self.tau)
            self.episode_trajectory[var] = []
        self.past_raw_trajectory.clear()

    def preprocess(self, tau_state, state):  # return pre-processed tau_mdp state
        tau_num = len(tau_state["battery"])
        # assert tau_num == self.tau, "dim of tau-state is wrong."

        obs = np.zeros(self.observation_space.shape[0])
        for i in range(self.pure_state_length):
            obs[i] = state[i]

        # ==== Always rules ====
        # if self.subSTL_1_robustness(tau_state[i]) >= 0:
        #     f1 = min(f1 + 1/(float(self.phi_1_timebound[1] - self.phi_1_timebound[0] + 1)), 1.0)
        # else:
        #     f1 = 0.0
        # ==== Always rules ====

        # ==== Eventually rules ====
        # if self.subSTL_1_robustness(tau_state[i]) >= 0:
        #     f1 = 1.0
        # else:
        #     f1 = max(f1 - 1/(float(self.tau_1 + 1)), 0.0)
        # ==== Eventually rules ====
        # reach_charger = 0
        # reach_goal = 0
        # for i in range(tau_num):
        #     # Always reach target
        #     if self.reach_target_robustness(tau_state["target_distance"][i]) >= 0:
        #         reach_goal = min(reach_goal + 1 / (float(self.tau + 1)), 1.0)
        #     else:
        #         reach_goal = 0.0

        #     # Eventually reach charger
        #     if self.reach_charger_robustness(tau_state["charger_distance"][i]) >= 0:
        #         reach_charger = 1.0
        #     else:
        #         reach_charger = max(reach_charger - 1 / (float(self.tau + 1)), 0.0)

        # obs[self.pure_state_length] = reach_goal - 0.5
        # obs[self.pure_state_length + 1] = reach_charger - 0.5

        (
            rho_charger,
            rho_avoid,
            rho_target,
            rho_battery,
            cost_battery,
            cost_target,
            cost_charger,
            cost_avoid,
            for_preprocess,
        ) = self.get_robustness()
        eventually_charger = for_preprocess["minimize_charger"]
        eventually_target = for_preprocess["minimize_target"]
        always_objective = for_preprocess["reach_charger_single"]

        eventually_charger_flag = 0
        eventually_goal_flag = 0
        always_objective_flag = 0
        for i in range(tau_num):
            # Eventually reach charger
            if eventually_charger[i] >= 0:
                eventually_charger_flag = 1.0
            else:
                eventually_charger_flag = max(
                    eventually_charger_flag - 1 / (float(self.tau + 1)), 0.0
                )

            # Eventually reach target
            if eventually_target[i] >= 0:
                eventually_goal_flag = 1.0
            else:
                eventually_goal_flag = max(
                    eventually_goal_flag - 1 / (float(self.tau + 1)), 0.0
                )

            # Always objective
            if always_objective[i] >= 0:
                always_objective_flag = min(
                    always_objective_flag + 1 / (float(self.tau + 1)), 1.0
                )
            else:
                always_objective_flag = 0.0

        # obs[self.pure_state_length] = rho_goal  # reach_goal - 0.5
        obs[self.pure_state_length] = eventually_charger_flag - 0.5
        obs[self.pure_state_length + 1] = eventually_goal_flag - 0.5
        obs[self.pure_state_length + 1] = always_objective_flag - 0.5

        return obs

    def compute_step(self, action):
        action = torch.from_numpy(action)
        # Clip action bundaries
        velocity = torch.clip(action[0], 0, 1)
        angle = torch.clip(action[1], -np.pi, np.pi)

        state = torch.from_numpy(self.state)
        next_state, new_pose = self.sim.update_state(
            state,
            velocity,
            angle,
            self.pose,
            self.obstacles,
            self.target,
            self.chargers,
        )

        # Update past trakectory
        next_state = next_state.cpu().numpy()
        self.append_to_traj_dict(next_state)

        # Collision check using the new robot pose

        collision = check_path_collision(
            self.obs, self.pose[:2], new_pose[:2], self.device
        )
        next_state = self.fix_state_for(
            next_state,
            new_pose,
            self.target,
            torch.tensor(next_state[-4:-2], device=self.device),
        )
        return state, next_state, collision

    def step(self, action):
        state, next_state, collision = self.compute_step(action)

        (
            rho_charger,
            rho_avoid,
            rho_goal,
            rho_battery,
            cost_battery,
            cost_goal,
            cost_charger,
            cost_avoid,
            _,
        ) = self.get_robustness()

        self.state = next_state
        self.num_steps += 1

        goal_reached = bool(self.state[8] < 0.08)
        battery = bool(self.state[11] <= 0.1)
        truncated = self.num_steps >= self.max_steps

        reward = self.reward(
            state, next_state, action, (goal_reached, collision, battery, truncated)
        )

        done = goal_reached or battery or collision or truncated

        lidar_min_mean = 0
        low_battery_corr = 0
        mean_battery = 0
        stay_at_charger = 0
        if done:
            self.episode_trajectory["time"] = range(
                len(self.episode_trajectory["battery"])
            )
            # Reeplace rho with total episode computation
            # rho_charger, rho_avoid, rho_goal, rho_battery, _, _, _, _ = (
            #     self.get_robustness(self.episode_trajectory)
            # )
            lidar_min_mean, low_battery_corr, mean_battery, stay_at_charger = (
                self.calc_metrics()
            )

        return_value = (
            self.state,
            reward,  #   + cost_goal,
            done,
            {
                "collision": collision,
                "goal_reached": goal_reached,
                "battery": battery,
                "truncated": truncated,
                "total_objective_reward": reward,  # + reward_goal,
                "reward_goal": cost_goal,
                "reward_charger": cost_charger,
                "cost_avoid": cost_avoid,
                "cost_battery": cost_battery,
                "rho_goal": rho_goal,
                "rho_avoid": rho_avoid,
                "rho_charger": rho_charger,
                "rho_battery": rho_battery,
                "lidar_mean": lidar_min_mean,
                "battery_corr": low_battery_corr,
                "mean_battery": mean_battery,
                "stay_at_charger": stay_at_charger,
            },
        )

        return return_value

    def calc_metrics(self):
        mean_min_lidar = np.mean(np.array(self.episode_trajectory["lidar_min"][10:]))

        # --- 2. Correlation: low battery (< 2) vs charger distance ---
        battery_values = np.array(self.episode_trajectory["battery"][10:])
        charger_distances = np.array(self.episode_trajectory["charger_distance"][10:])
        
        def checkNan(value: float):
            return 0 if np.isnan(value) else value
            
        mean_dist_low = checkNan(np.mean(charger_distances[battery_values < 0.6]))
        mean_dist_high = checkNan(np.mean(charger_distances[battery_values >= 0.6]))
        lb_compliance = checkNan(np.mean(charger_distances[battery_values < 0.6] < self.close_enough_charger))

        return (
            mean_min_lidar,
            (mean_dist_low, mean_dist_high, lb_compliance),
            np.mean(battery_values),
            np.sum(charger_distances < self.close_enough_charger),
        )

    def build_stl(self):
        stl_spec = rtamt.STLDiscreteTimeSpecification()
        stl_spec.name = "STL safety"

        # Variables declaration
        stl_spec.declare_var("battery", "float")
        stl_spec.declare_var("charger_distance", "float")
        stl_spec.declare_var("target_distance", "float")
        stl_spec.declare_var("lidar_min", "float")
           

        # Const declaration
        stl_spec.declare_const("min_battery", "float", self.minimum_battery / 5)
        stl_spec.declare_const("critic_battery", "float", 1 / 5)
        stl_spec.declare_const("safe_battery", "float", 3 / 5)
        stl_spec.declare_const("close_enough", "float", self.close_enough_charger)
        stl_spec.declare_const("safe_distance", "float", self.safe_distance)
        stl_spec.declare_const(
            "close_enough_target", "float", self.close_enough_target
        )

        # Subformula declaration
        stl_spec.declare_var("distance", "float")
        stl_spec.declare_var("reach_charger", "float")
        stl_spec.declare_var("safe_behavior", "float")
        stl_spec.declare_var("reach_target", "float")
        stl_spec.declare_var("high_battery", "float")
        stl_spec.declare_var("minimize_charger", "float")
        stl_spec.declare_var("minimize_target", "float")

        stl_spec.add_sub_spec("minimize_charger = charger_distance < close_enough")

        stl_spec.add_sub_spec("minimize_target = lidar_min > safe_distance")

        # stl_spec.add_sub_spec(
        #     "reach_charger_single = (battery < min_battery) -> (eventually(minimize_charger)) and (battery >= min_battery) -> (eventually(minimize_target))"
        # )
        # and (battery >= min_battery) -> eventually(minimize_target)
        stl_spec.add_sub_spec(
            "reach_charger_single = minimize_charger until battery > safe_battery"
        )
        stl_spec.add_sub_spec("reach_charger = always(reach_charger_single)")

        stl_spec.add_sub_spec(
            "reach_target = always( (battery >= min_battery) -> eventually(minimize_target))"
        )
        stl_spec.add_sub_spec("high_battery = always(battery > min_battery)")
        stl_spec.add_sub_spec(
            "safe_behavior = always(minimize_target)"
        )

        stl_spec.spec = (
            "reach_charger and safe_behavior and reach_target and high_battery"
        )

        try:
            stl_spec.parse()
        except rtamt.RTAMTException as err:
            print("RTAMT Exception: {}".format(err))
            sys.exit()

        return stl_spec

    def get_robustness(self, trajectory=None):
        # print(trajectory)
        # print(self.past_tau_trajectory)
        self.rules.evaluate(
            self.past_tau_trajectory if trajectory is None else trajectory
        )
        rho_charger = self.rules.get_value("reach_charger")[0]
        rho_avoid = self.rules.get_value("safe_behavior")[0]
        rho_target = self.rules.get_value("reach_target")[0]
        rho_battery = self.rules.get_value("high_battery")[0]

        for_preprocess = {
            "minimize_charger": self.rules.get_value("minimize_charger"),
            "minimize_target": self.rules.get_value("minimize_target"),
            "reach_charger_single": self.rules.get_value("reach_charger_single"),
        }

        # cost_avoid = - np.exp(-self.beta * int(rho_avoid >= 0))
        # cost_charger =  - np.exp(-self.beta * int(rho_charger >= 0))
        # cost_target = - np.exp(-self.beta * int(rho_target >= 0))
        # cost_battery = - np.exp(-self.beta * int(rho_battery >= 0))

        # cost_avoid =  1 / (1 + np.exp(-rho_avoid)) if rho_avoid < 0 else 0
        # cost_charger =  1 / (1 + np.exp(-rho_charger)) if rho_charger < 0 else 0
        # cost_target =  1 / (1 + np.exp(-rho_target)) if rho_target < 0 else 0
        # cost_battery =  1 / (1 + np.exp(-rho_battery)) if rho_battery < 0 else 0

        cost_avoid = np.tanh(rho_avoid) 
        cost_charger = np.tanh(rho_charger) 
        cost_target = np.tanh(rho_target) 
        cost_battery = np.tanh(rho_battery) 

        return (
            rho_charger,
            rho_avoid,
            rho_target,
            rho_battery,
            cost_battery,
            cost_target,
            cost_charger,
            cost_avoid,
            for_preprocess,
        )

    def append_to_traj_dict(
        self,
        state,
    ):
        self.past_tau_trajectory["lidar_min"].append(min(state[0:7]))
        self.episode_trajectory["lidar_min"].append(min(state[0:7]))

        self.past_tau_trajectory["target_distance"].append(state[8])
        self.episode_trajectory["target_distance"].append(state[8])
        self.past_tau_trajectory["charger_distance"].append(state[10])
        self.episode_trajectory["charger_distance"].append(state[10])
        self.past_tau_trajectory["battery"].append(state[11] / 5)
        self.episode_trajectory["battery"].append(state[11] / 5)

        self.past_raw_trajectory.append(state)

    def reward(self, state, next_state, action, ending):
        goal_reached, collision, battery, truncated = ending
        dg, dg_prev = next_state[8], state[8]  # distance to goal

        # Terminal rewards
        # if collision or battery or truncated:
        #     return -100

        # if truncated:
        #     return -1000

        if goal_reached:
            return 100

        return (dg_prev - dg) * 10

        # Progress reward (encourages moving toward goal)
        progress_reward = (dg_prev - dg) * 1000
        step_penalty = -0.1  # small penalty for each step

        return progress_reward + step_penalty

    # Eventually
    def reach_charger_robustness(self, distance):
        # charger_distance < close_enough
        return self.close_enough_charger - distance

    # Eventually
    def reach_target_robustness(self, distance):
        # charger_distance < close_enough
        return self.close_enough_target - distance


    def render(self):
        args = self.args
        nt = args.nt
        ti = self.num_steps
        bloat = 0.5

        fig, ax_list = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

        # --- Top Plot: Rover Environment ---
        ax = ax_list[0]
        self.plot_env(ax)

        # Complete state
        # print(self.complete_state)
        # [
        #    0.04672469  0.47685334  0.1978927   0.08945225  0.09296738  0.11010709 0.2479678  LIDAR
        #   -2.8317482  0.6998208                   (head, dist) TARGET RELATIVE
        #    2.81173    0.11152058                  (head, dist) CHARGER RELATIVE
        #    4.200001   0.4                         BATTERY
        #    9.829277    6.404966   -0.34621185     ROBOT POSE
        #    2.8356962   6.659418                   TARGET POSITION
        #    8.774196    6.766196                   CHARGER POSITION
        # ]
        state = self.complete_state
        path = np.array(self.history)

        # Key extracted values
        target_angle = state[7]
        charger_angle = state[9]
        rover_x, rover_y = state[-7], state[-6]
        goal_x, goal_y = state[-4], state[-3]
        theta = state[-5]
        lidar_ranges = state[:7]
        # print(lidar_ranges)

        # Robot, target, and charger
        ax.scatter(rover_x, rover_y, color="blue", label="Rover", s=50)
        ax.scatter(goal_x, goal_y, color="green", label="Destination", s=50)

        for charger in self.chargers.squeeze().unsqueeze(0).cpu().numpy():
            c_x, c_y = charger[0], charger[1]
            ax.scatter(c_x, c_y, color="gold", label="Charger", s=50)

        # Path trace and battery text
        ax.plot(path[:, 0], path[:, 1], color="blue", linewidth=2, alpha=0.6, zorder=10)
        for x, y, battery in zip(path[:, 0], path[:, 1], path[:, 6]):
            ax.text(x + 0.2, y + 0.2, f"{battery:.1f}", fontsize=6, alpha=0.7)

        # --- LIDAR Beams ---
        beam_angles = self.sim.beam_angles.cpu().numpy()
        max_range = self.sim.max_range_lidar

        for i, beam_angle in enumerate(beam_angles):
            norm_dist = lidar_ranges[i]  # normalized [0, 1]
            dist = norm_dist * max_range  # scale back to real-world distance
            end_x = rover_x + dist * np.cos(theta + beam_angle)
            end_y = rover_y + dist * np.sin(theta + beam_angle)
            ax.plot(
                [rover_x, end_x], [rover_y, end_y], color="red", linewidth=1, alpha=0.6
            )
            ax.scatter(end_x, end_y, color="red", s=10, zorder=15)  # debug dot
            ax.text(
                end_x + 0.1,
                end_y + 0.1,
                f"{norm_dist:.3f}",
                fontsize=6,
                color="red",
                alpha=0.7,
            )

        # --- Robot Orientation ---
        orient_len = 0.5
        head_x = rover_x + orient_len * np.cos(theta)
        head_y = rover_y + orient_len * np.sin(theta)
        ax.arrow(
            rover_x,
            rover_y,
            head_x - rover_x,
            head_y - rover_y,
            head_width=0.2,
            head_length=0.2,
            fc="blue",
            ec="blue",
        )
        # --- Target Angle Arrow ---
        target_arrow_len = 1.0
        target_end_x = rover_x + target_arrow_len * np.cos(target_angle)
        target_end_y = rover_y + target_arrow_len * np.sin(target_angle)
        ax.arrow(
            rover_x,
            rover_y,
            target_end_x - rover_x,
            target_end_y - rover_y,
            head_width=0.15,
            head_length=0.2,
            fc="green",
            ec="green",
            alpha=0.7,
            linestyle="--",
        )

        # --- Charger Angle Arrow ---
        charger_arrow_len = 1.0
        charger_end_x = rover_x + charger_arrow_len * np.cos(charger_angle)
        charger_end_y = rover_y + charger_arrow_len * np.sin(charger_angle)
        ax.arrow(
            rover_x,
            rover_y,
            charger_end_x - rover_x,
            charger_end_y - rover_y,
            head_width=0.15,
            head_length=0.2,
            fc="gold",
            ec="gold",
            alpha=0.7,
            linestyle="--",
        )

        ax.set_xlim(-bloat, 10 + bloat)
        ax.set_ylim(-bloat, 10 + bloat)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Rover Position", fontsize=12)

        # --- Bottom Plot: Metrics Over Time ---
        ax = ax_list[1]
        t_ranges = np.arange(len(path))
        s6 = path[:, 6] / (25 * args.dt)
        s7 = path[:, 7] / (25 * args.dt)
        at_goal = (
            np.linalg.norm(path[:, 0:2] - path[:, 2:4], axis=-1)
            < self.close_enough_target
        )
        at_charger = (
            np.linalg.norm(path[:, 0:2] - path[:, 4:6], axis=-1)
            < self.close_enough_charger
        )

        ax.plot(t_ranges, s6, label="Battery (%)", color="orange", linewidth=2)
        ax.plot(t_ranges, s7, label="Stay Time", color="blue", linewidth=2)
        ax.plot(
            t_ranges,
            at_charger.astype(float),
            label="At Charger",
            color="purple",
            linewidth=1.5,
            linestyle="--",
        )
        ax.plot(
            t_ranges,
            at_goal.astype(float),
            label="At Goal",
            color="gray",
            linewidth=1.5,
            linestyle=":",
        )

        ax.set_ylim(-0.2, 1.2)
        ax.set_xticks(t_ranges[:: max(1, len(t_ranges) // 10)])
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Normalized Values", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"Simulation Metrics ({ti}/{nt})", fontsize=12)

        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()

    def plot_env(self, ax):
        for obj in self.obstacles:
            if isinstance(obj, dict):
                cx, cy = obj["center"]
                w, h = (
                    obj["width"] - (self.sim.epsilon * 2),
                    obj["height"] - (self.sim.epsilon * 2),
                )
            else:
                cx, cy, w, h = (
                    obj[0].item(),
                    obj[1].item(),
                    obj[2].item() - (self.sim.epsilon * 2),
                    obj[3].item() - (self.sim.epsilon * 2),
                )
            lower_left = (cx - w / 2, cy - h / 2)
            rect_patch = patches.Rectangle(
                lower_left,
                w,
                h,
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                alpha=0.5,
            )
            ax.add_patch(rect_patch)
