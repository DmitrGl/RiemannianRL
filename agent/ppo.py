import numpy as np
import torch


class RolloutBuffer:

    def __init__(self, n, gamma = 0.99, lam = 0.95):
        self.n = n
        self.cur = 0
        self.local_states = []
        self.global_states = []
        self.ego_features = []
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
        self.ego_features = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        self.cur = 0

    def store(self, local_state, global_state, ego_feature, action, logprob, reward, value, is_terminal):
        i = self.cur
        self.local_states.append(local_state)
        self.global_states.append(global_state)
        self.ego_features.append(ego_feature)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.is_terminals.append(is_terminal)
        self.cur += 1

class PPOAgent:

    def __init__(self, actor, critic, lr_actor=3e-4, lr_critic=1e-3, K_epochs=10, eps_clip=0.2, batch_size = 64, entropy_coef = 0.01, target_kl = 0.015, gamma=0.99, total_updates = 1000):
        self.actor = actor
        self.critic = critic
        self.lr_a = lr_actor
        self.lr_c = lr_critic
        self.eps = eps_clip
        self.gamma = gamma
        self.K = K_epochs
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr= lr_actor, eps= 1e-5)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr= lr_critic, eps= 1e-5)
        lr_lambda = lambda step: max(0.0, 1.0 - step / total_updates)
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optim, lr_lambda)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optim, lr_lambda)
        self.batch_size = batch_size
        self.local_rms = RunningMeanStd()
        self.global_rms = RunningMeanStd()
        self.ego_rms = RunningMeanStd()
        self.ret_rms = RunningMeanStd()
        self._disc_return = 0.0
        self.is_training = True 

    def set_training(self, flag: bool):
        self.is_training = flag


    def select_action(self, local_obs, global_obs, ego_features):
        if(self.local_rms.mean is None):
            self.local_rms.mean = np.zeros_like(local_obs)
            self.global_rms.mean = np.zeros_like(global_obs)
            self.ego_rms.mean = np.zeros_like(ego_features)
        n_local, n_global, n_ego = self.normalize_obs(local_obs, global_obs, ego_features)


        obs = torch.tensor(np.concatenate((n_local, n_global, n_ego)), dtype=torch.float32)
        local = torch.tensor(n_local, dtype=torch.float32)
        
        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(local)
        return action.numpy(), action_logprob.item(), state_val.item(), n_local, n_global, n_ego

    def update(self, buffer, next_value):
        advantages, returns = buffer.compute_gae(next_value)

        
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean)/(adv_std + 1e-8)
        
        global_states = torch.tensor(np.asarray(buffer.global_states, dtype=np.float32), dtype=torch.float32)
        local_states = torch.tensor(np.asarray(buffer.local_states, dtype=np.float32), dtype=torch.float32)
        ego_features = torch.tensor(np.asarray(buffer.ego_features, dtype=np.float32), dtype=torch.float32)
        states = torch.cat([local_states,global_states,ego_features], dim = 1)
        actions = torch.tensor(np.asarray(buffer.actions, dtype=np.float32), dtype=torch.float32)
        logprobs = torch.tensor(np.asarray(buffer.logprobs, dtype=np.float32), dtype=torch.float32)
        values = torch.tensor(np.asarray(buffer.values, dtype=np.float32), dtype=torch.float32)

        actor_losses = []
        critic_losses = []
        approx_kls = []
        clip_fractions = []
        entropies = []
        explained_variances = []

        
        for _ in range(self.K):
            indices = torch.randperm(buffer.n)
            e = self.eps
            kls = []

            for batch_start in range(0, buffer.n, self.batch_size):
                batch_end = batch_start + self.batch_size
                batch = indices[batch_start : batch_end]
                
                mb_states = states[batch]
                mb_local_states = local_states[batch]
                mb_actions = actions[batch]

                mb_logprobs = logprobs[batch]
                mb_advantages = advantages[batch]
                
                mb_returns = returns[batch]
                mb_new_logprobs = self.actor(mb_states).log_prob(mb_actions)
                mb_new_values = self.critic(mb_local_states).squeeze(-1)
                mb_values = values[batch]
                entropy = -mb_new_logprobs.mean().item()

                r = torch.exp(mb_new_logprobs - mb_logprobs)
                #v_clipped = mb_values + torch.clamp(mb_new_values - mb_values, -e, e)
                loss_actor = -torch.mean(torch.minimum(mb_advantages*r, torch.clamp(r,1-e,1+e)*mb_advantages))
                loss_actor = loss_actor - self.entropy_coef*entropy 
                loss_critic = torch.mean((mb_new_values - mb_returns)**2)
                #loss_critic = torch.mean(torch.maximum((mb_new_values - mb_returns)**2,(v_clipped - mb_returns)**2))

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                loss_actor.backward()
                loss_critic.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
    
                self.actor_optim.step()
                self.critic_optim.step()
                

                with torch.no_grad():
                    approx_kl = ((r-1) - (mb_new_logprobs-mb_logprobs)).mean().item()
                    clip_fraction =((r - 1.0).abs() > e).float().mean().item()

                    kls.append(approx_kl)
                    
                    var_returns = torch.var(mb_returns, unbiased=False)
                    var_residual = torch.var(mb_returns - mb_new_values, unbiased=False)
                    explained_variance = 1.0 - (var_residual / (var_returns + 1e-8)).item()
                    actor_losses.append(loss_actor.item())
                    critic_losses.append(loss_critic.item())
                    approx_kls.append(approx_kl)
                    clip_fractions.append(clip_fraction)
                    entropies.append(entropy)
                    explained_variances.append(explained_variance)
            if(np.mean(kls) > self.target_kl):
                break
        
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        metrics = {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "policy_entropy": float(np.mean(entropies)),
            "explained_variance": float(np.mean(explained_variances)),
            "mean_advantage": adv_mean.item(),
            "std_advantage": adv_std.item(),
            "mean_return": returns.mean().item(),
            "std_return": returns.std().item(),
            "policy/log_std_0": self.actor.log_std[0].item(),
            "policy/log_std_1": self.actor.log_std[1].item(),
        }
        return metrics

    def normalize_obs(self, local_obs, global_obs=None, ego_features=None, update=None):
        update = self.is_training if update is None else update

        if update:
            self.local_rms.update(local_obs)
        n_local = self.local_rms.normalize(local_obs)

        if global_obs is None:
            return n_local

        if update:
            self.global_rms.update(global_obs)
            self.ego_rms.update(ego_features)
        n_global = self.global_rms.normalize(global_obs)
        n_ego = self.ego_rms.normalize(ego_features)

        return n_local, n_global, n_ego
    
    def normalize_reward(self, reward, done=False, update=None):
        update = self.is_training if update is None else update
        if(self.ret_rms.mean is None):
            self.ret_rms.mean = np.zeros_like(self._disc_return)
        if update:
            self._disc_return = self._disc_return * self.gamma + reward
            self.ret_rms.update(self._disc_return)
            if done:
                self._disc_return = 0.0

        return self.ret_rms.scale(reward)

    def get_next_value(self, local_obs, done=False):
        if done:
            return 0.0
        n_local = self.normalize_obs(local_obs, update=False)
        with torch.no_grad():
            return float(self.critic(torch.tensor(n_local, dtype=torch.float32)).squeeze(-1).item())

class RunningMeanStd:

    def __init__(self, eps=1e-8):
        self.mean = None
        self.var = None
        self.count = 0
        self.eps = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.mean is not None and x.shape == self.mean.shape:
            x = x[np.newaxis, ...]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        if batch_count == 0:
            return

        if self.mean is None:               
            self.mean = np.zeros_like(batch_mean)
        if(self.var is None):
            self.var = np.ones_like(batch_var)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = (x - self.mean) / np.sqrt(self.var + self.eps)
        return y.astype(np.float32)

    def scale(self, x):
        y = np.asarray(x, dtype=np.float64) / np.sqrt(self.var + self.eps)
        return y.astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"].copy()
        self.var = d["var"].copy()
        self.count = d["count"]