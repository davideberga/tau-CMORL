from copy import deepcopy
import random
from algorithms.trpo import CR_MOGM, Cagrad_all, trpo_step
from utils.utils import (
    get_flat_grad_from,
    get_flat_params_from,
    normal_log_density,
    set_flat_params_to,
)
from algorithms.models import Policy, Value
import torch
from torch.autograd import Variable
import numpy as np
import scipy
from rich import print


class AgentPendulum:
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        max_kl: float = 0.01,
        start_safety: int = 100,
        safety_bound: float = 0.3, 
        gamma: float = 0.99,
        tau: float = 0.95,
        l2_reg: float = 1e-3,
        damping: float = 1e-1,
        level_of_conflict : float = 0.5, # Balanced
        env = None
    ):
        # Parameters
        self.gamma = gamma
        self.tau = tau
        self.l2_reg = l2_reg
        self.damping = damping
        self.max_kl = max_kl
        self.start_safety = start_safety
        self.safety_bound = safety_bound
        self.level_of_conflict = level_of_conflict
        self.env = env

        # Networks
        self.policy_net = Policy(state_size, action_size)
        self.value_objective_net = Value(state_size)

        self.cost_net_thetadot = Value(state_size)
        self.cost_net_torque = Value(state_size)

    def select_action(self, state, no_grads: bool = False):
        state = torch.from_numpy(state).unsqueeze(0).double()
        action_mean, _, action_std = self.policy_net(Variable(state))
        action = torch.normal(action_mean, action_std)
        
        # Squashed
        action_tanh = torch.tanh(action)
        
        if self.env is not None:
            low = torch.tensor(self.env.action_space.low, dtype=torch.float32)
            high = torch.tensor(self.env.action_space.high, dtype=torch.float32)
            action_tanh = low + (0.5 * (action_tanh + 1.0) * (high - low))
        
        return action.cpu(), action_tanh.cpu()
        # state = torch.from_numpy(state).unsqueeze(0)
        # if no_grads:
        #     with torch.no_grad:
        #         action_mean, _, action_std = self.policy_net(Variable(state))
        # else:
        #     action_mean, _, action_std = self.policy_net(Variable(state))
        # action = torch.normal(action_mean, action_std)
        # return action.cpu().detach()
        
    def compute_advantage(self, actions, rewards, masks, values):
        returns = torch.Tensor(actions.size(0), 1)
        deltas = torch.Tensor(actions.size(0), 1)
        advantages = torch.Tensor(actions.size(0), 1)
        
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]  # A = Q-V
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
            
        return advantages, Variable(returns)

    def update_params(self, batch, rho_thetadot, rho_torque, i_episode):
        costs_thetadot = torch.Tensor(batch.cost_thetadot)
        costs_torque = torch.Tensor(batch.cost_torque)
        rewards = torch.Tensor(batch.reward)

        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)

        values = self.value_objective_net(Variable(states))
        values_cost_thetadot = self.cost_net_thetadot(Variable(states))
        values_cost_torque = self.cost_net_torque(Variable(states))
        
        advantages, targets = self.compute_advantage(actions, rewards, masks, values)
        advantages_cost_thetadot, targets_cost_thetadot = self.compute_advantage(actions, costs_thetadot, masks, values_cost_thetadot)
        advantages_cost_torque, targets_cost_torque = self.compute_advantage(actions, costs_torque, masks, values_cost_torque)

        # Original code uses the same LBFGS to optimize the value loss
        def get_cost_loss_thetadot(flat_params):
            set_flat_params_to(self.cost_net_thetadot, torch.Tensor(flat_params))
            for param in self.cost_net_thetadot.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            costs_ = self.cost_net_thetadot(Variable(states))

            cost_loss = (costs_ - targets_cost_thetadot).pow(2).mean()

            # weight decay
            for param in self.cost_net_thetadot.parameters():
                cost_loss += param.pow(2).sum() * self.l2_reg
            cost_loss.backward()
            return (cost_loss.data.double().numpy(), get_flat_grad_from(self.cost_net_thetadot).data.double().numpy())
        
        def get_cost_loss_torque(flat_params):
            set_flat_params_to(self.cost_net_torque, torch.Tensor(flat_params))
            for param in self.cost_net_torque.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            costs_ = self.cost_net_torque(Variable(states))

            cost_loss = (costs_ - targets_cost_torque).pow(2).mean()

            # weight decay
            for param in self.cost_net_torque.parameters():
                cost_loss += param.pow(2).sum() * self.l2_reg
            cost_loss.backward()
            return (cost_loss.data.double().numpy(), get_flat_grad_from(self.cost_net_torque).data.double().numpy())

        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            set_flat_params_to(self.value_objective_net, torch.Tensor(flat_params))
            for param in self.value_objective_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.value_objective_net(Variable(states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in self.value_objective_net.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_flat_grad_from(self.value_objective_net).data.double().numpy())

            # Original code uses the same LBFGS to optimize the value loss

        if (rho_thetadot >= 0 and rho_torque >= 0) or i_episode <= self.start_safety:
            
            # if i_episode > self.start_safety:
            #     print("[bold green] Working [/bold green]")
                
            flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                                    get_flat_params_from(self.value_objective_net).double().numpy(),
                                                                    maxiter=25)
            set_flat_params_to(self.value_objective_net, torch.Tensor(flat_params))

            advantages = (advantages - advantages.mean()) / advantages.std()

            action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
            fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

            def get_loss(volatile=False):
                if volatile:
                    with torch.no_grad():
                        action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                else:
                    action_means, action_log_stds, action_stds = self.policy_net(Variable(states))

                log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
                return action_loss.mean()

            def get_kl():
                mean1, log_std1, std1 = self.policy_net(Variable(states))

                mean0 = Variable(mean1.data)
                log_std0 = Variable(log_std1.data)
                std0 = Variable(std1.data)
                kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                return kl.sum(1, keepdim=True)

            trpo_step(self.policy_net, get_loss, get_kl, self.max_kl, self.damping)
            
        else:
            
            print(f"[bold red] Constraint violated {rho_thetadot} >= 0 and {rho_torque} >= 0 [/bold red]")
            
            to_upgrade = [] # ["TH", "TO"]
            if rho_thetadot < 0: to_upgrade.append("TH")
            if rho_torque < 0: to_upgrade.append("TO")
            
            if len(to_upgrade) == 1: 
                choice = to_upgrade[0]
                
                if choice == "TH":
                    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_cost_loss_thetadot,
                                                                        get_flat_params_from(self.cost_net_thetadot).double().numpy(),
                                                                        maxiter=25)
                    set_flat_params_to(self.cost_net_thetadot, torch.Tensor(flat_params))
                    advantages_cost = (advantages_cost_thetadot - advantages_cost_thetadot.mean()) / advantages_cost_thetadot.std()
                    action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
                else: 
                    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_cost_loss_torque,
                                                                        get_flat_params_from(self.cost_net_torque).double().numpy(),
                                                                        maxiter=25)
                    set_flat_params_to(self.cost_net_torque, torch.Tensor(flat_params))
                    advantages_cost = (advantages_cost_torque - advantages_cost_torque.mean()) / advantages_cost_torque.std()
                    action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

                def get_cost_loss(volatile=False):
                    if volatile:
                        with torch.no_grad():
                            action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    else:
                        action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                    action_loss = -Variable(advantages_cost) * torch.exp(log_prob - Variable(fixed_log_prob))
                    return action_loss.mean()

                def get_kl():
                    mean1, log_std1, std1 = self.policy_net(Variable(states))
                    mean0 = Variable(mean1.data)
                    log_std0 = Variable(log_std1.data)
                    std0 = Variable(std1.data)
                    kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                    return kl.sum(1, keepdim=True)

                trpo_step(self.policy_net, get_cost_loss, get_kl, self.max_kl, self.damping)
            else:
                
                print("CAGRAD update")
                flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_cost_loss_thetadot,
                                                                        get_flat_params_from(self.cost_net_thetadot).double().numpy(),
                                                                        maxiter=25)
                set_flat_params_to(self.cost_net_thetadot, torch.Tensor(flat_params))
               
                
                flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_cost_loss_torque,
                                                                        get_flat_params_from(self.cost_net_torque).double().numpy(),
                                                                        maxiter=25)
                set_flat_params_to(self.cost_net_torque, torch.Tensor(flat_params))
                
                advantages_cost_thetadot = (advantages_cost_thetadot - advantages_cost_thetadot.mean()) / advantages_cost_thetadot.std()
                advantages_cost_torque = (advantages_cost_torque - advantages_cost_torque.mean()) / advantages_cost_torque.std()
                
                action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
                
                def get_cost_loss_thetadot(volatile=False):
                    if volatile:
                        with torch.no_grad():
                            action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    else:
                        action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                    action_loss = -Variable(advantages_cost_thetadot) * torch.exp(log_prob - Variable(fixed_log_prob))
                    return action_loss.mean()
                
                def get_cost_loss_torque(volatile=False):
                    if volatile:
                        with torch.no_grad():
                            action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    else:
                        action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                    action_loss = -Variable(advantages_cost_torque) * torch.exp(log_prob - Variable(fixed_log_prob))
                    return action_loss.mean()
                
                def get_kl():
                    mean1, log_std1, std1 = self.policy_net(Variable(states))
                    mean0 = Variable(mean1.data)
                    log_std0 = Variable(log_std1.data)
                    std0 = Variable(std1.data)
                    kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                    return kl.sum(1, keepdim=True)
                
                prev_policy_net = deepcopy(self.policy_net)
                prev_policy_net_data = get_flat_params_from(prev_policy_net)
                
                
                trpo_step(self.policy_net, get_cost_loss_thetadot, get_kl, self.max_kl, self.damping)
                grads1 = get_flat_params_from(self.policy_net) - prev_policy_net_data
                set_flat_params_to(self.policy_net, prev_policy_net_data)
                
                trpo_step(self.policy_net, get_cost_loss_torque, get_kl, self.max_kl, self.damping)
                grads2 = get_flat_params_from(self.policy_net) - prev_policy_net_data

                
                cagrad_all_tasks = Cagrad_all(c=0.5)
                add_dem_grads1 = grads1.unsqueeze(0)
                add_dem_grads2 = grads2.unsqueeze(0)
                grad_vec = torch.cat((add_dem_grads1, add_dem_grads2))
                final_grad = cagrad_all_tasks.cagrad(grad_vec, len(grad_vec))
                set_flat_params_to(self.policy_net, prev_policy_net_data + final_grad)
                

        # Return if contraints is violated
        return rho_thetadot >= 0 and rho_torque >= 0