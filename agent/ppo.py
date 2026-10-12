import numpy as np
import torch


class RolloutBuffer:

    def __init__(self, n, gamma = 0.99, lam = 0.95):
        self.n = n
        self.cur = 0
        self.local_states = []
        self.global_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        self.gamma = gamma
        self.lam = lam
    
    def compute_gae(self, next_value):
        gamma = self.gamma
        lam = self.lam
        values = np.array(self.values)
        delta_t = self.rewards[-1] + gamma*next_value*(1-self.is_terminals[-1]) - values[-1]
        advantages = np.zeros(self.n)
        for t in reversed(range(self.n)):
            if t == (self.n - 1):
                advantages[t] = delta_t
            else:
                delta_t = self.rewards[t] + gamma*values[t+1]*(1-self.is_terminals[t]) - values[t]
                advantages[t] = delta_t + gamma*lam*(1-self.is_terminals[t])*advantages[t+1]
        returns = advantages + values
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def clear(self):
        self.local_states = []
        self.global_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        self.cur = 0

    def store(self, local_state, global_state, action, logprob, reward, value, is_terminal):
        i = self.cur
        self.local_states.append(local_state)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.is_terminals.append(is_terminal)
        self.cur += 1

class PPOAgent:

    def __init__(self, actor, critic, lr_actor=3e-4, lr_critic=1e-3, K_epochs=10, eps_clip=0.2):
        self.actor = actor
        self.critic = critic
        self.lr_a = lr_actor
        self.lr_c = lr_critic
        self.eps = eps_clip
        self.K = K_epochs
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr_actor)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr_critic)

    def select_action(self, local_obs, global_obs):
        obs = torch.tensor(np.concat((local_obs,global_obs)), dtype=torch.float32)
        local = torch.tensor(local_obs, dtype=torch.float32)
        
        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(local)
        return action.numpy(), action_logprob.item(), state_val.item()

    def update(self, buffer, next_value):
        advantages, returns = buffer.compute_gae(next_value)
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        
        global_states = torch.tensor(np.asarray(buffer.global_states, dtype=np.float32), dtype=torch.float32)
        local_states = torch.tensor(np.asarray(buffer.local_states, dtype=np.float32), dtype=torch.float32)
        states = torch.cat([local_states,global_states], dim = 1)
        actions = torch.tensor(np.asarray(buffer.actions, dtype=np.float32), dtype=torch.float32)
        logprobs = torch.tensor(np.asarray(buffer.logprobs, dtype=np.float32), dtype=torch.float32)
        
        for _ in range(self.K):
            new_logprobs = self.actor(states).log_prob(actions)
            new_values = self.critic(local_states).squeeze()
            r = torch.exp(new_logprobs - logprobs)
            e = self.eps
            loss_actor = -torch.mean(torch.minimum(advantages*r, torch.clamp(r,1-e,1+e)*advantages))
            loss_critic = torch.mean((new_values - returns)**2)
            
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()