from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy
import time
from pathlib import Path


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class DDPGAgent(BaseAgent):
    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'ddpg'
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr = self.cfg.lr
        self.pi = Policy(state_dim, self.action_dim,
                         self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(
            self.pi.parameters(), lr=float(self.lr))
        # self.buffer = ReplayBuffer(state_dim[0], self.action_dim, max_size=int(float(self.buffer_size)))

        self.batch_size = self.cfg.batch_size
        self.buffer_size = self.cfg.buffer_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        self.buffer = ReplayBuffer(state_shape=[
                                   state_dim], action_dim=self.action_dim, max_size=int(float(self.cfg.buffer_size)))

        self.q = Critic(state_dim, self.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(
            self.q.parameters(), lr=float(self.lr))

        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0
        self.random_transition = 5000  # collect 5k random data for better exploration
        self.max_episode_steps = self.cfg.max_episode_steps
    # Update the critic network
    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head

        if self.buffer_ptr > self.random_transition:  # update once we have enough data
            for _ in range(update_iter):
                info = self._update()

        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info

    # 1. compute target Q, you should not modify the gradient of the variables

    def calculate_target(self, batch):
        ########## Your code starts here. ##########
        with torch.no_grad():
            next_action = self.pi_target(batch.next_state)
            q_tar = self.q_target(batch.next_state, next_action)
            target_Q = batch.reward + self.gamma * batch.not_done * q_tar
        ########## Your code ends here. ##########
        return target_Q
    # 2. compute critic loss

    def calculate_critic_loss(self, current_Q, target_Q):
        ########## Your code starts here. ##########
        critic_loss = F.mse_loss(current_Q, target_Q)
        ########## Your code ends here. ##########
        return critic_loss
    # 3. compute actor loss

    def calculate_actor_loss(self, batch):
        ########## Your code starts here. ##########
        actions = self.pi(batch.state)
        actor_loss = -self.q(batch.state, actions).mean()
        ########## Your code ends here. ##########
        return actor_loss

    def _update(self):
        """ Update the pi and q networks. """
        # Sample a batch of transitions from the buffer
        batch = self.buffer.sample(self.batch_size, self.device)

        current_Q = self.q(batch.state, batch.action)
        target_Q = self.calculate_target(batch)

        critic_loss = self.calculate_critic_loss(current_Q, target_Q)

        # optimize the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        actor_loss = self.calculate_actor_loss(batch)

        # optimize the actor
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        cu.soft_update_params(self.q, self.q_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {}

    # @torch.no_grad()
    # def get_action(self, observation, evaluation=False):
    #     # add the batch dimension
    #     if observation.ndim == 1:
    #         observation = observation[None]
    #     try:
    #         x = torch.from_numpy(observation).float().to(self.device)
    #     except:
    #         x = observation

    #     if self.buffer_ptr < self.random_transition:
    #         action = torch.rand(self.action_dim)
    #     else:
    #         expl_noise = 0.1 * self.max_action

    #         action = self.pi(x)

    #         if not evaluation:
    #             # action = (action + expl_noise *
    #             #             torch.randn_like(action)).clamp(-self.max_action, self.max_action)
    #             noises = torch.normal(
    #                 0, expl_noise, size=action.size()).to(self.device)
    #             action = action + noises
    #             action = action.clamp(-self.max_action, self.max_action)
    #     return action, {}
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if isinstance(observation, list):
            observation = np.array(observation)  # Ensure observation is a numpy array
        if observation.ndim == 1:
            observation = np.expand_dims(observation, axis=0)  # Add batch dimension if necessary
        
        x = torch.from_numpy(observation).float().to(self.device)

        if not evaluation:
            expl_noise = 0.1 * self.max_action
            action = (self.pi(x) + expl_noise * torch.randn_like(self.pi(x))).clamp(-self.max_action, self.max_action)
        else:
            action = self.pi(x).clamp(-self.max_action, self.max_action)

        return action, {}
    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    def train_iteration(self):
        # start = time.perf_counter()
        # Run actual training
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:

            # Sample action from policy
            action, act_logprob = self.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)

            done_bool = float(
                done) if timesteps < self.max_episode_steps else 0
            self.record(obs, action, next_obs, reward, done_bool)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        # s = time.perf_counter()
        info = self.update()
        # e = time.perf_counter()

        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })

        end = time.perf_counter()
        return info

    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger()  # create a simple logger to record stats
        start = time.perf_counter()
        total_step = 0
        run_episode_reward = []
        log_count = 0

        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({'episodes': ep})
            total_step += train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])

            if total_step > self.cfg.log_interval*log_count:
                average_return = sum(run_episode_reward) / \
                    len(run_episode_reward)
                if not self.cfg.silent:
                    print(
                        f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}")
                if self.cfg.save_logging:
                    train_info.update({'average_return': average_return})
                    L.log(**train_info)
                run_episode_reward = []
                log_count += 1

        if self.cfg.save_model:
            self.save_model()

        logging_path = str(self.logging_dir)+'/logs'
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end-start)/60
        print('------ Training Finished ------')
        print(f'Total traning time is {train_time}mins')

    def load_model(self):
        # define the save path, do not modify
        filepath = str(self.model_dir) + '/model_parameters_' + str(self.seed) + '.pt'
        print(f'Loading model: {filepath}')
        d = torch.load(filepath)
        self.q.load_state_dict(d['q'])
        self.q_target.load_state_dict(d['q_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])

    def save_model(self):
        # define the save path, do not modify
        filepath = str(self.model_dir)+'/model_parameters_' + \
            str(self.seed)+'.pt'

        torch.save({
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")
