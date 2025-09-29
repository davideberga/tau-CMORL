from copy import deepcopy
import gymnasium
import torch
import numpy as np
from collections import deque

from agent_pendulum import AgentPendulum
from algorithms.trpo import CR_MOGM, trpo_step
from algorithms.models import Policy, Value
from algorithms.replay_memory_O2_2_costs import Memory2Costs
from env.Pendulum import Pendulum
from utils.running_state import ZFilter
from rich import print
import random

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
safeties = [50, 100, 150]
seeds_to_inspect = [ 42, 43, 44, 45, 46]

# NN 128

for START_SAFETY in safeties:
    for use_this_seed in seeds_to_inspect:
        
        seed_everything(use_this_seed)
        
        # Use Pendulum
        gymnasium.register(
            id="Pendulum-vBerga",
            entry_point=Pendulum,
            max_episode_steps=200,  # Prevent infinite episodes
        )

        # Use Pendulum
        env = gymnasium.make("Pendulum-vBerga")
        # env_test = gymnasium.make("Pendulum-vBerga")



        # torch.set_default_device(device)
        torch.set_default_dtype(torch.double)
        # torch.manual_seed(use_this_seed)
        # np.random.seed(use_this_seed)
        
        
        
        


        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        # Hyperparameters
        # GAMMA = 0.995
        # TAU = 0.97
        # L2_REG = 1e-3
        # DAMPING = 1e-1
        # EPISODE_LENGTH = 200
        # EPISODE_PER_BATCH = 50
        # MAX_EPISODES = 1000
        # LOG_INTERVAL = 20
        # MAX_KL = 0.001
        
        GAMMA = 0.97
        TAU = 0.95
        L2_REG = 1e-3
        DAMPING = 1e-1
        EPISODE_LENGTH = 200
        EPISODE_PER_BATCH = 30
        MAX_EPISODES = 300
        LOG_INTERVAL = 20
        MAX_KL = 0.005

        agent = AgentPendulum(
            num_inputs,
            num_actions,
            max_kl=MAX_KL,
            safety_bound=-0.4,
            start_safety=START_SAFETY, # 120,
            damping=DAMPING,
            gamma=GAMMA,
            l2_reg=L2_REG,
            tau=TAU,
            level_of_conflict=0.0
        )
        
        seed_everything(use_this_seed)
        
        print(f"[bold green]Using start_safety {START_SAFETY} seed: {use_this_seed} [/bold green]")

        running_state = ZFilter((num_inputs,))

        # === Main training loop ===
        all_rewards = []
        all_eval_returns = []
        perc_satified = []
        all_theta = []
        all_thetadot = []
        all_torque = []
        
        all_costs_thetadot = []
        all_costs_torque = []
        
        all_rho_thetadot = []
        all_rho_torque = []

        reward_list_100 = deque(maxlen=100)
        cost_list_100 = deque(maxlen=100)
        state = env.reset(seed=use_this_seed)[0]

        for i_episode in range(1, MAX_EPISODES + 1):
            memory = Memory2Costs()
            num_steps = 0
            reward_batch = 0
            satisfied_list = []
            
            episode_theta = []
            episode_thetadot = []
            episode_torque = []
            
            episode_cost_thetadot = []
            episode_cost_torque = []
            
            episode_rho_thetadot = []
            episode_rho_torque = []
            
            
            while num_steps < EPISODE_LENGTH * EPISODE_PER_BATCH:
                state = running_state(env.reset()[0])

                episode_reward = 0
                episode_cost = 0
            
                for t in range(EPISODE_LENGTH):

                    action,  act_squashed = agent.select_action(state)
                    action, act_squashed = action.detach().numpy().squeeze(0), act_squashed.detach().numpy().squeeze(0)
                    next_state, reward, done, truncated, info = env.step(action)
                    next_state = running_state(next_state)
                    
                    cost_thetadot = info["cost_thetadot"] * 100
                    cost_torque = info["cost_torque"] * 100
                    episode_cost = cost_thetadot + cost_torque
                    
                    rho_thetadot = info["rho_thetadot"]
                    rho_torque = info["rho_torque"]
                    
                    theta = info["theta"]
                    thetadot = info["thetadot"]
                    torque = info["torque"]
                    
                    
                    
                    mask = 0 if t == EPISODE_LENGTH - 1 else 1
                    memory.push(state, np.array([action]), mask, next_state, reward, cost_thetadot, cost_torque)
                    
                    episode_reward += reward
                    
                    episode_theta.append(theta)
                    episode_thetadot.append(thetadot)
                    episode_torque.append(torque)
                    episode_cost_thetadot.append(cost_thetadot)
                    episode_cost_torque.append(cost_torque)
                    episode_rho_thetadot.append(rho_thetadot)
                    episode_rho_torque.append(rho_torque)
            
                    
                    state = next_state
                    if done or truncated:
                        break
                    
                num_steps += EPISODE_LENGTH
                reward_batch += episode_reward
                # cost_step += episode_cost
                
                reward_list_100.append(episode_reward)
                cost_list_100.append(episode_cost)
                
            
            
            
            batch = memory.sample()
            satisfied = agent.update_params(batch, np.mean(episode_rho_thetadot), np.mean(episode_rho_torque), i_episode)
            satisfied_list.append(1 if satisfied else 0)
            eval_return = 0 # evaluate(env_test, episodes=10)
                
            mean_rew = np.mean(reward_list_100)
            mean_cost = np.mean(cost_list_100)
            sat= np.mean(satisfied_list)
            
            all_rewards.append(mean_rew)

            all_theta.append(np.mean(episode_theta))
            all_thetadot.append(np.mean(episode_thetadot))
            all_torque.append(np.mean(episode_torque))
        
            all_costs_thetadot.append(np.mean(episode_cost_thetadot))
            all_costs_torque.append(np.mean(episode_cost_torque))
            
            all_rho_thetadot.append(np.mean(episode_rho_thetadot))
            all_rho_torque.append(np.mean(episode_rho_torque))

            perc_satified.append(sat)
            
            print(f"Episode {i_episode}, Last 100 Avg Reward: {mean_rew:.2f}, Last 100 Avg Cost: {-mean_cost:.2f}, Episodes satisfied contraints {sat:.2f}")
            all_eval_returns.append(eval_return)

            if i_episode % LOG_INTERVAL == 0:
                pass
                

        # Save rewards
        np.savez(f"results/pendulum/our_safety_{START_SAFETY}_{use_this_seed}.npz", 
                rewards=np.array(all_rewards),
                sat=np.array(perc_satified),
                theta=np.array(all_theta),
                thetadot=np.array(all_thetadot),
                torque=np.array(all_torque),
                cost_thetadot=np.array(all_costs_thetadot),
                cost_torque=np.array(all_costs_torque),
                rho_thetadot=np.array(all_rho_thetadot),
                rho_torque=np.array(all_rho_torque) 
        )
