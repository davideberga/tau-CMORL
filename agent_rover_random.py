from copy import deepcopy
import random
from algorithms.trpo import Cagrad_all, trpo_step
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


class AgentRoverRandom:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        max_kl: float = 0.01,
        start_safety: int = 100,
        gamma: float = 0.99,
        tau: float = 0.95,
        l2_reg: float = 1e-3,
        damping: float = 1e-1,
        level_of_conflict: float = 0.5,  # Balanced
        env=None,
    ):
        # Parameters
        self.gamma = gamma
        self.tau = tau
        self.l2_reg = l2_reg
        self.damping = damping
        self.max_kl = max_kl
        self.start_safety = start_safety
        self.level_of_conflict = level_of_conflict
        self.env = env

        # Networks
        self.policy_net = Policy(state_size, action_size)
        self.objective_net = Value(state_size)

        self.cost_net_avoid = Value(state_size)
        self.cost_net_charger = Value(state_size)

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

    def compute_advantage(self, actions, rewards, masks, values):
        returns = torch.Tensor(actions.size(0), 1)
        deltas = torch.Tensor(actions.size(0), 1)
        advantages = torch.Tensor(actions.size(0), 1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = (
                rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]
            )  # A = Q-V
            advantages[i] = (
                deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]
            )

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        return advantages, Variable(returns)

    @staticmethod
    def get_cost_loss(flat_params, nn, states, targets_costs, l2_reg):
        set_flat_params_to(nn, torch.Tensor(flat_params))
        for param in nn.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        costs_ = nn(Variable(states))
        cost_loss = (costs_ - targets_costs).pow(2).mean()

        # weight decay
        for param in nn.parameters():
            cost_loss += param.pow(2).sum() * l2_reg
        cost_loss.backward()
        return (
            cost_loss.data.double().numpy(),
            get_flat_grad_from(nn).data.double().numpy(),
        )

    def update_params(self, batch, rho_charger, rho_avoid, i_episode):
        print(f"[bold blue]   AVOID: {rho_avoid}  BATTERY: {rho_charger} [/bold blue]")
        costs_avoid = torch.Tensor(batch.cost_avoid)
        costs_charger = torch.Tensor(batch.cost_charger)
        rewards = torch.Tensor(batch.reward)

        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)

        values = self.objective_net(Variable(states))
        values_cost_avoid = self.cost_net_avoid(Variable(states))
        values_cost_charger = self.cost_net_charger(Variable(states))

        advantages, targets = self.compute_advantage(actions, rewards, masks, values)
        advantages_cost_avoid, targets_cost_avoid = self.compute_advantage(
            actions, costs_avoid, masks, values_cost_avoid
        )
        advantages_cost_charger, targets_cost_charger = self.compute_advantage(
            actions, costs_charger, masks, values_cost_charger
        )

        def get_value_loss(flat_params):
            return AgentRoverRandom.get_cost_loss(
                flat_params, self.objective_net, states, targets, self.l2_reg
            )

        def get_cost_loss_avoid(flat_params):
            return AgentRoverRandom.get_cost_loss(
                flat_params,
                self.cost_net_avoid,
                states,
                targets_cost_avoid,
                self.l2_reg,
            )

        def get_cost_loss_charger(flat_params):
            return AgentRoverRandom.get_cost_loss(
                flat_params,
                self.cost_net_charger,
                states,
                targets_cost_charger,
                self.l2_reg,
            )

        if (rho_avoid > 0 and rho_charger > 0) or i_episode <= self.start_safety:
            # Cost are satisfied

            flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
                get_value_loss,
                get_flat_params_from(self.objective_net).double().numpy(),
                maxiter=25,
            )
            set_flat_params_to(self.objective_net, torch.Tensor(flat_params))

            # Z-score on advantages
            advantages = (advantages - advantages.mean()) / advantages.std()

            action_means, action_log_stds, action_stds = self.policy_net(
                Variable(states)
            )
            fixed_log_prob = normal_log_density(
                Variable(actions), action_means, action_log_stds, action_stds
            ).data.clone()

            def get_loss(volatile=False):
                if volatile:
                    with torch.no_grad():
                        action_means, action_log_stds, action_stds = self.policy_net(
                            Variable(states)
                        )
                else:
                    action_means, action_log_stds, action_stds = self.policy_net(
                        Variable(states)
                    )

                log_prob = normal_log_density(
                    Variable(actions), action_means, action_log_stds, action_stds
                )
                action_loss = -Variable(advantages) * torch.exp(
                    log_prob - Variable(fixed_log_prob)
                )
                return action_loss.mean()

            def get_kl():
                mean1, log_std1, std1 = self.policy_net(Variable(states))

                mean0 = Variable(mean1.data)
                log_std0 = Variable(log_std1.data)
                std0 = Variable(std1.data)
                kl = (
                    log_std1
                    - log_std0
                    + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))
                    - 0.5
                )
                return kl.sum(1, keepdim=True)

            trpo_step(self.policy_net, get_loss, get_kl, self.max_kl, self.damping)

        else:
            print(
                f"[bold red] AVOID: {rho_avoid} and CHARGER: {rho_charger} [/bold red]"
            )

            to_upgrade = []  # ["AVOID", "CHARGER"]
            if rho_avoid < 0:
                to_upgrade.append("AVOID")
            if rho_charger < 0:
                to_upgrade.append("CHARGER")
            choice = random.choice(to_upgrade)

            if choice == "AVOID":
                flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
                    get_cost_loss_avoid,
                    get_flat_params_from(self.cost_net_avoid).double().numpy(),
                    maxiter=25,
                )
                set_flat_params_to(self.cost_net_avoid, torch.Tensor(flat_params))
                advantages_cost = (
                    advantages_cost_avoid - advantages_cost_avoid.mean()
                ) / advantages_cost_avoid.std()
                action_means, action_log_stds, action_stds = self.policy_net(
                    Variable(states)
                )
                fixed_log_prob = normal_log_density(
                    Variable(actions), action_means, action_log_stds, action_stds
                ).data.clone()
            else:
                flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
                    get_cost_loss_charger,
                    get_flat_params_from(self.cost_net_charger).double().numpy(),
                    maxiter=25,
                )
                set_flat_params_to(self.cost_net_charger, torch.Tensor(flat_params))
                advantages_cost = (
                    advantages_cost_charger - advantages_cost_charger.mean()
                ) / advantages_cost_charger.std()
                action_means, action_log_stds, action_stds = self.policy_net(
                    Variable(states)
                )
                fixed_log_prob = normal_log_density(
                    Variable(actions), action_means, action_log_stds, action_stds
                ).data.clone()

            def get_cost_loss(volatile=False):
                if volatile:
                    with torch.no_grad():
                        action_means, action_log_stds, action_stds = (
                            self.policy_net(Variable(states))
                        )
                else:
                    action_means, action_log_stds, action_stds = self.policy_net(
                        Variable(states)
                    )
                log_prob = normal_log_density(
                    Variable(actions), action_means, action_log_stds, action_stds
                )
                action_loss = -Variable(advantages_cost) * torch.exp(
                    log_prob - Variable(fixed_log_prob)
                )
                return action_loss.mean()

            def get_kl():
                mean1, log_std1, std1 = self.policy_net(Variable(states))
                mean0 = Variable(mean1.data)
                log_std0 = Variable(log_std1.data)
                std0 = Variable(std1.data)
                kl = (
                    log_std1
                    - log_std0
                    + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))
                    - 0.5
                )
                return kl.sum(1, keepdim=True)

            trpo_step(
                self.policy_net, get_cost_loss, get_kl, self.max_kl, self.damping
            )
            

        # Return if contraints are violated
        return rho_avoid >= 0 and rho_charger >= 0
