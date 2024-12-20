from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy
import time
from pathlib import Path


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, state_shape, action_dim, max_size=int(1e6), alpha=0.6):
        super().__init__(state_shape, action_dim, max_size)
        self.alpha = alpha
        self.priorities = np.zeros((max_size,), dtype=np.float32)

    def add(self, state, action, next_state, reward, done, extra=None):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        super().add(state, action, next_state, reward, done, extra)
        self.priorities[self.ptr] = max_priority

    def sample(self, batch_size, beta=0.4, device='cpu'):
        beta = float(beta)  # Ensure beta is float
        if self.size == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = super().sample(batch_size, device)
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority


class DDPGExtensionFurther(DDPGAgent):
    def __init__(self, config=None):
        super(DDPGExtensionFurther, self).__init__(config)
        self.sil_alpha = getattr(
            self.cfg, 'sil_alpha', 0.1)  # Default SIL alpha
        self.sil_beta = getattr(self.cfg, 'sil_beta',
                                0.1)    # Default SIL beta

    def calculate_sil_loss(self, batch):
        with torch.no_grad():
            next_action = self.pi_target(batch.next_state)
            target_Q = batch.reward + self.gamma * batch.not_done * \
                self.q_target(batch.next_state, next_action)

        current_Q = self.q(batch.state, batch.action)
        advantage = target_Q - current_Q

        sil_loss = F.mse_loss(current_Q, target_Q) + \
            self.sil_alpha * F.relu(-advantage).mean()
        return sil_loss

    def update(self):
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head

        if self.buffer_ptr > self.random_transition:
            for _ in range(update_iter):
                batch = self.buffer.sample(self.batch_size, self.device)

                current_Q = self.q(batch.state, batch.action)
                target_Q = self.calculate_target(batch)

                critic_loss = self.calculate_critic_loss(current_Q, target_Q)
                sil_loss = self.calculate_sil_loss(batch)

                total_critic_loss = critic_loss + self.sil_beta * sil_loss

                self.q_optim.zero_grad()
                total_critic_loss.backward()
                self.q_optim.step()

                actor_loss = self.calculate_actor_loss(batch)

                self.pi_optim.zero_grad()
                actor_loss.backward()
                self.pi_optim.step()

                cu.soft_update_params(self.q, self.q_target, self.tau)
                cu.soft_update_params(self.pi, self.pi_target, self.tau)

        self.buffer_head = self.buffer_ptr
        return info
