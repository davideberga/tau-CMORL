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
    all_costs = []

    all_rho_avoid = []
    all_rho_charger = []

    all_end_goal = []
    all_end_collision = []
    all_end_battery = []
    all_end_truncated = []

    all_lidar_mean = []
    all_battery_mean = []
    all_stay_at_charger = []

    reward_list_100 = deque(maxlen=100)
    cost_100 = deque(maxlen=100)

    state = env.reset()

    for i_episode in range(1, MAX_EPISODES + 1):
        memory = Memory()  #  RoverMemory6() # RoverMemoryNoObjective()
        num_steps = 0
        reward_batch = 0

        episode_rho_avoid = []
        episode_rho_charger = []
        episodes_ending = []

        episodes_lidar = []
        episodes_mean_battery = []
        episodes_stay_at_charger = []

        trajs = []

        while num_steps < STEPS_PER_BATCH:
            state = env.reset()
            
            episode_reward = []
            episode_total_cost = []

            while True:
                # Do action
                action, act_squashed = agent.select_action(state)
                action, act_squashed = (
                    action.detach().numpy().squeeze(0),
                    act_squashed.detach().numpy().squeeze(0),
                )
                next_state, reward, done, info = env.step(action)
                num_steps += 1

                cost_avoid = info["cost_avoid"]
                cost_charger = info["cost_charger"]

                episode_cost = cost_avoid + cost_charger

                mask = 1
                if done:
                    mask = 0
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
                    0,
                    cost_avoid,
                    cost_charger,
                )

                episode_reward.append(reward)
                episode_total_cost.append(episode_cost)

                episode_rho_avoid.append(info["rho_avoid"])
                episode_rho_charger.append(info["rho_charger"])

                if done:
                    # Calc metrics for entire episode
                    episodes_lidar.append(info["lidar_mean"])
                    episodes_mean_battery.append(info["mean_battery"])
                    episodes_stay_at_charger.append(info["stay_at_charger"])

                    break

                state = next_state

            reward_list_100.append(np.sum(episode_reward))
            cost_100.append(np.sum(episode_cost))

        batch = memory.sample()
        satisfied = agent.update_params(
            batch,
            np.mean(episode_rho_charger),
            np.mean(episode_rho_avoid),
            i_episode,
        )
    
        # Return and total cost
        mean_rew = np.mean(reward_list_100)
        mean_cost = -np.mean(cost_100)
        all_costs.append(mean_cost)
        all_rewards.append(mean_rew)

        # Qualitatives averages
        mean_lid = np.mean(episodes_lidar)
        mean_battery = np.mean(episodes_mean_battery)
        mean_stay = np.mean(episodes_stay_at_charger)
        all_lidar_mean.append(mean_lid)
        all_battery_mean.append(mean_battery)
        all_stay_at_charger.append(mean_stay)

        # Qunatitative satisfaction
        all_rho_avoid.append(np.mean(episode_rho_avoid))
        all_rho_charger.append(np.mean(episode_rho_charger))

        # Episode endings
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
            f"Batch {i_episode}, Rw: {mean_rew:.2f}, Cost: {mean_cost:.2f}, G: {perc_goal}, C: {perc_collision}, B: {perc_battery}, T: {perc_trunc}, E: {n_epi}, L: {np.round(mean_lid, 3)}, MB: {np.round(mean_battery, 3)}, Stay: {np.round(mean_stay, 3)}"
        )

    np.savez(
        f"results/rover/random_safety_{START_SAFETY}_{seed_to_use}.npz",
        rewards=np.array(all_rewards),
        all_costs=np.array(all_costs),
        mean_lidar=np.array(all_lidar_mean),
        mean_battery=np.array(all_battery_mean),
        rho_avoid=np.array(all_rho_avoid),
        rho_charger=np.array(all_rho_charger),
        mean_charger_time=np.array(all_stay_at_charger),
        goal=all_end_goal,
        collision=all_end_collision,
        battery=all_end_battery,
        truncated=all_end_truncated,
    )
