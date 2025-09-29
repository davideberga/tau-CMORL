from collections import deque
import random
import torch
import numpy as np

from agent_rover_random import AgentRoverRandom
from algorithms.rover_memory import Memory
from env.RoverEnvSafety import RoverEnvCMORLSafety, ENV_MODE, STATE_MODE
from utils.running_state import ZFilter
from config.config import get_config

from rich import print

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_dtype(torch.double)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: for PyTorch >= 1.8, for stricter determinism
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Not available in older versions


seeds = [42, 43, 44, 45, 46]

START_SAFETY = 70

for seed_to_use in seeds:
    print(
        f"[bold green] Safety: {START_SAFETY},  Using seed {seed_to_use} [/bold green]"
    )
    seed_everything(seed_to_use)

    args = get_config().parse_args()
    env = RoverEnvCMORLSafety(
        args,
        STATE_MODE.RELAXATION,
        ENV_MODE.NORMAL,
        True,
        max_steps=50,
        time_horizon=10,
    )
    # env_test = RoverEnvCMORLSparse(args, STATE_MODE.RELAXATION, ENV_MODE.NORMAL, False)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # env.seed(args.seed)
    # torch.manual_seed(args.seed)

    running_state = ZFilter((num_inputs,))

    # Hyperparameters
    GAMMA = 0.99
    TAU = 0.97
    L2_REG = 1e-3
    DAMPING = 1e-1
    STEPS_PER_BATCH = 5000
    MAX_EPISODES = 150
    LOG_INTERVAL = 20
    MAX_KL = 0.01

    # GAMMA = 0.995
    # TAU = 0.97
    # L2_REG = 1e-3
    # DAMPING = 1e-1
    # STEPS_PER_BATCH = 1500
    # MAX_EPISODES = 500
    # LOG_INTERVAL = 20
    # MAX_KL = 0.02

    agent = AgentRoverRandom(
        num_inputs,
        num_actions,
        max_kl=MAX_KL,
        start_safety=START_SAFETY,
        damping=DAMPING,
        gamma=GAMMA,
        l2_reg=L2_REG,
        tau=TAU,
        level_of_conflict=0.5,
    )

    all_rewards = []
    all_eval_returns = []
    perc_satified = []

    all_reward_sparse = []
    all_reward_goal = []
    all_reward_charger = []

    all_costs_avoid = []
    all_costs_battery = []
    all_costs = []

    all_rho_goal = []
    all_rho_avoid = []
    all_rho_charger = []
    all_rho_battery = []

    all_end_goal = []
    all_end_collision = []
    all_end_battery = []
    all_end_truncated = []

    all_lidar_mean = []
    all_battery_corr = []
    all_battery_mean = []
    all_stay_at_charger = []
    all_mean_dist_low = []
    all_mean_dist_high = []
    all_lb_compliance = []

    reward_list_100 = deque(maxlen=100)
    reward_sparse_list_100 = deque(maxlen=100)
    reward_goal_100 = deque(maxlen=100)
    reward_charger_100 = deque(maxlen=100)

    cost_100 = deque(maxlen=100)

    state = env.reset()

    for i_episode in range(1, MAX_EPISODES + 1):
        memory = Memory()  #  RoverMemory6() # RoverMemoryNoObjective()
        num_steps = 0
        reward_batch = 0
        satisfied_list = []

        episode_reward_sparse = []
        episode_reward_goal = []
        episode_reward_charger = []

        episode_cost_avoid = []
        episode_cost_battery = []

        episode_rho_goal = []
        episode_rho_avoid = []
        episode_rho_charger = []
        episode_rho_battery = []
        episodes_ending = []

        episodes_lidar = []
        episodes_battery_corr = []
        episodes_mean_battery = []
        episodes_stay_at_charger = []

        episodes_mean_dist_low = []
        episodes_mean_dist_high = []
        episodes_lb_compliance = []

        trajs = []

        while num_steps < STEPS_PER_BATCH:
            state = env.reset()
            # state = running_state(state)

            episode_sparse = 0
            episode_goal = 0
            episode_charger = 0

            episode_reward = 0
            episode_cost = 0

            while True:
                # Do action
                action, act_squashed = agent.select_action(state)
                action, act_squashed = (
                    action.detach().numpy().squeeze(0),
                    act_squashed.detach().numpy().squeeze(0),
                )
                next_state, reward, done, info = env.step(action)
                # next_state = running_state(next_state)
                num_steps += 1

                reward_goal, reward_charger = (
                    info["reward_goal"],
                    info["reward_charger"],
                )
                # reward_goal = info["reward_goal"]
                # reward_charger = info["reward_charger"]
                # reward_charger += reward
                # reward_goal += reward

                cost_avoid = info["cost_avoid"]
                cost_battery = info["cost_battery"]

                episode_cost = cost_avoid + reward_charger

                rho_goal = info["rho_goal"]

                mask = 1
                if done:
                    mask = 0
                    # if info["truncated"]:
                    #     reward_goal += -1000
                    #     reward_charger += -1000
                    # if info["collision"] or info["battery"]:
                    #     reward_goal += -500
                    #     reward_charger += -500
                    episodes_ending.append(
                        np.array(
                            [
                                info["goal_reached"],
                                info["collision"],
                                info["battery"],
                                info["truncated"],
                            ]
                        )
                    )

                memory.push(
                    state,
                    np.array([action]),
                    mask,
                    next_state,
                    reward,
                    reward_charger,
                    cost_avoid,
                    reward_charger,
                )

                episode_reward += reward
                episode_sparse += reward
                episode_goal += reward_goal
                episode_charger += reward_charger

                episode_reward_sparse.append(reward)
                episode_reward_goal.append(reward_goal)
                episode_reward_charger.append(reward_charger)

                episode_cost_avoid.append(cost_avoid)
                episode_cost_battery.append(cost_battery)
                episode_rho_battery.append(info["rho_battery"])
                episode_rho_avoid.append(info["rho_avoid"])
                episode_rho_charger.append(info["rho_charger"])
                episode_rho_goal.append(info["rho_goal"])

                if done:
                    # Calc metrics for entire episode
                    episodes_lidar.append(info["lidar_mean"])
                    # episodes_battery_corr.append(info["battery_corr"])
                    episodes_mean_battery.append(info["mean_battery"])
                    episodes_stay_at_charger.append(info["stay_at_charger"])

                    mean_dist_low, mean_dist_high, lb_compliance = info[
                        "battery_corr"
                    ]
                    episodes_mean_dist_low.append(mean_dist_low)
                    episodes_mean_dist_high.append(mean_dist_high)
                    episodes_lb_compliance.append(lb_compliance)
                    # trajs.append(env.episode_trajectory)

                    # print(f"RHO AVOID {info['rho_avoid']}")
                    # print(f"RHO CHARGER {info['rho_charger']}")
                    # print(f"RHO GOAL {info['rho_goal']}")
                    # env.render()

                    break

                state = next_state

            reward_list_100.append(episode_reward)
            reward_sparse_list_100.append(episode_sparse)
            reward_goal_100.append(episode_goal)
            reward_charger_100.append(episode_charger)
            cost_100.append(episode_cost)

        # np.savez(
        #     "results/cmorl/train/trajectory.npz",
        #     trajectories=np.array(trajs)
        # )

        # exit(0)

        batch = memory.sample()
        satisfied = agent.update_params(
            batch,
            np.mean(episode_rho_charger),
            np.mean(episode_rho_avoid),
            i_episode,
        )
        satisfied_list.append(1 if satisfied else 0)
        eval_return = 0  # evaluate(env_test, episodes=10)

        mean_rew = np.mean(reward_list_100)
        mean_rew_sparse = np.mean(reward_sparse_list_100)
        mean_rew_goal = np.mean(reward_goal_100)
        mean_rew_charger = np.mean(reward_charger_100)
        mean_cost = -np.mean(cost_100)
        sat = np.mean(satisfied_list)

        mean_lid = np.mean(episodes_lidar)
        mean_bat_cor = np.mean(episodes_battery_corr)
        mean_battery = np.mean(episodes_mean_battery)
        mean_stay = np.mean(episodes_stay_at_charger)

        mean_mean_dist_low = np.mean(episodes_mean_dist_low)
        mean_mean_dist_high = np.mean(episodes_mean_dist_high)
        mean_lb_compliance = np.mean(episodes_lb_compliance)

        all_costs.append(mean_cost)

        all_lidar_mean.append(mean_lid)
        all_battery_corr.append(mean_bat_cor)
        all_battery_mean.append(mean_battery)
        all_stay_at_charger.append(mean_stay)

        all_mean_dist_low.append(mean_mean_dist_low)
        all_mean_dist_high.append(mean_mean_dist_high)
        all_lb_compliance.append(mean_lb_compliance)

        all_rewards.append(mean_rew)
        all_reward_sparse.append(mean_rew_sparse)
        all_reward_goal.append(mean_rew_goal)
        all_reward_charger.append(mean_rew_charger)

        all_costs_avoid.append(np.mean(episode_cost_avoid))
        all_costs_battery.append(np.mean(episode_cost_battery))

        all_rho_goal.append(np.mean(episode_rho_goal))
        all_rho_avoid.append(np.mean(episode_rho_avoid))
        all_rho_charger.append(np.mean(episode_rho_charger))
        all_rho_battery.append(np.mean(episode_rho_battery))

        perc_satified.append(sat)

        n_epi = len(episodes_ending)
        ending = np.array(episodes_ending)
        perc_goal = np.round(np.sum(ending[:, 0]) / n_epi, 2)
        perc_collision = np.round(np.sum(ending[:, 1]) / n_epi, 2)
        perc_battery = np.round(np.sum(ending[:, 2]) / n_epi, 2)
        perc_trunc = np.round(np.sum(ending[:, 3]) / n_epi, 2)

        all_end_goal.append(perc_goal)
        all_end_collision.append(perc_collision)
        all_end_battery.append(perc_battery)
        all_end_truncated.append(perc_trunc)

        print(
            f"Batch {i_episode}, Rw: {mean_rew_sparse:.2f}, RwG: {mean_rew_goal:.2f}, RwC: {mean_rew_charger:.2f} Cost: {mean_cost:.2f}, G: {perc_goal}, C: {perc_collision}, B: {perc_battery}, T: {perc_trunc}, E: {n_epi}, L: {np.round(mean_lid, 3)}, MB: {np.round(mean_battery, 3)}, Stay: {np.round(mean_stay, 3)}, MDL: {np.round(mean_mean_dist_low, 3)}, MDH: {np.round(mean_mean_dist_high, 3)}, Compliance: {np.round(mean_lb_compliance, 3)}"
        )
        all_eval_returns.append(eval_return)

    np.savez(
        f"results/rover/random_safety_{START_SAFETY}_{seed_to_use}.npz",
        rewards=np.array(all_rewards),
        sat=np.array(perc_satified),
        cost_avoid=np.array(all_costs_avoid),
        cost_charger=np.array(all_reward_charger),
        cost_battery=np.array(all_costs_battery),
        all_costs=np.array(all_costs),
        mean_lidar=np.array(all_lidar_mean),
        mean_battery=np.array(all_battery_mean),
        mean_charger_time=np.array(all_stay_at_charger),
        battery_corr=np.array(all_battery_corr),
        all_mean_dist_low=np.array(all_mean_dist_low),
        all_mean_dist_high=np.array(all_mean_dist_high),
        all_lb_compliance=np.array(all_lb_compliance),
        goal=all_end_goal,
        collision=all_end_collision,
        battery=all_end_battery,
        truncated=all_end_truncated,
        rho_goal=np.array(all_rho_goal),
        rho_avoid=np.array(all_rho_avoid),
        rho_charger=np.array(all_rho_charger),
        rho_battery=np.array(all_rho_battery),
    )
