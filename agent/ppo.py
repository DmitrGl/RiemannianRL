"""
PPO (clip) with GAE(lambda), written from scratch.

Contents:
    RolloutBuffer  - stores one rollout and computes GAE advantages / returns
    PPOAgent       - action selection, observation normalization and the PPO update
    RunningMeanStd - online mean / variance estimator used for normalization

Implementation details worth knowing (see also the project README):
    - actor and critic have different inputs: actor sees [local, global, ego], critic sees [local, ego]
    - local and global observations are normalized with running statistics, ego features are used as is
    - the buffer stores the already normalized observations that the networks actually saw
    - advantages are normalized over the whole rollout, returns are computed before that
    - the update stops early when the mean approximate KL of an epoch exceeds `target_kl`
    - gradient clipping (0.5), Adam eps=1e-5, linear learning-rate decay, small entropy bonus
"""
import numpy as np
import torch


class RolloutBuffer:
    """
    Fixed-size storage for one on-policy rollout of `n` steps.
    Everything is kept in Python lists and converted to tensors in PPOAgent.update().
    """

    def __init__(self, n, gamma=0.99, lam=0.95):
        self.n = n
        self.cur = 0  # number of stored steps
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
        """
        Generalized Advantage Estimation, computed backwards over the rollout:

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

        `done_t` cuts the trace at episode boundaries. `next_value` is V of the state that follows
        the last stored step (bootstrap for a rollout that ends in the middle of an episode).
        Expects a full buffer (len == n).

        Returns: (advantages, returns) as float32 tensors, with returns = advantages + values
        (not normalized).
        """
        gamma = self.gamma
        lam = self.lam
        values = np.array(self.values)

        # The last step is special: its successor value comes from `next_value`, not from the buffer.
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
        """Drop all stored steps (call after every update)."""
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
        """
        Append one transition. Observations must be exactly what was fed to the networks
        (i.e. already normalized); `action` is the raw sampled action (before tanh).
        """
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
    """PPO agent: owns the actor, the critic, their optimizers and the normalization statistics."""

    def __init__(
        self,
        actor,
        critic,
        lr_actor=3e-4,
        lr_critic=1e-3,
        K_epochs=10,
        eps_clip=0.2,
        batch_size=64,
        entropy_coef=0.01,
        target_kl=0.015,
        gamma=0.99,
        total_updates=1000,
    ):
        self.actor = actor
        self.critic = critic
        self.lr_a = lr_actor
        self.lr_c = lr_critic
        self.eps = eps_clip  # PPO clipping range
        self.gamma = gamma
        self.K = K_epochs  # max number of epochs per update
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl  # early-stopping threshold for the mean approximate KL

        # Separate optimizers for actor and critic; eps=1e-5 is the usual PPO setting.
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=lr_critic, eps=1e-5)

        # Linear learning-rate decay to zero over `total_updates` PPO updates
        # (the schedulers are stepped once per update() call).
        lr_lambda = lambda step: max(0.0, 1.0 - step / total_updates)
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optim, lr_lambda)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optim, lr_lambda)
        self.batch_size = batch_size

        # Running statistics for observation normalization.
        self.local_rms = RunningMeanStd()
        self.global_rms = RunningMeanStd()
        # Optional reward normalization (see normalize_reward): statistics of the discounted return.
        self.ret_rms = RunningMeanStd()
        self._disc_return = 0.0
        # When False, normalization statistics are frozen (inference / evaluation).
        self.is_training = True

    def set_training(self, flag: bool):
        """Enable / disable updates of the normalization statistics."""
        self.is_training = flag

    def select_action(self, local_obs, global_obs, ego_features):
        """
        Sample an action for a single (raw) observation.

        Returns: (action, logprob, value, n_local, n_global, n_ego)
            action   - raw action sampled from the Gaussian policy (numpy, NOT squashed by tanh)
            logprob  - log-probability of that raw action
            value    - critic estimate V(s)
            n_*      - the observations as the networks saw them (normalized); store these in the buffer
        """
        # RunningMeanStd.update() treats a 1-D input as a single sample only when `mean`
        # already exists, so initialize the means with zeros on the first call.
        if self.local_rms.mean is None:
            self.local_rms.mean = np.zeros_like(local_obs)
            self.global_rms.mean = np.zeros_like(global_obs)

        # Updates the statistics (in training mode) and normalizes. Ego features pass through unchanged.
        n_local, n_global, n_ego = self.normalize_obs(local_obs, global_obs, ego_features)
        n_ego = ego_features

        # Actor sees everything (26 values), critic only local + ego (14 values).
        obs = torch.tensor(np.concatenate((n_local, n_global, n_ego)), dtype=torch.float32)
        critic_obs = torch.tensor(np.concatenate((n_local, n_ego)), dtype=torch.float32)

        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(critic_obs)
        return action.numpy(), action_logprob.item(), state_val.item(), n_local, n_global, n_ego

    def update(self, buffer, next_value):
        """
        Run the PPO update on a full rollout buffer and return a dict of training diagnostics.

        Steps: GAE -> advantage normalization -> up to K epochs of minibatch updates with the
        clipped surrogate objective (early stop on KL) -> learning-rate decay step.
        """
        # --- advantages and returns -------------------------------------------------------------
        advantages, returns = buffer.compute_gae(next_value)

        # Normalize advantages over the whole rollout (returns were computed before this).
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean)/(adv_std + 1e-8)

        # --- rollout buffer -> tensors ----------------------------------------------------------
        global_states = torch.tensor(np.asarray(buffer.global_states, dtype=np.float32), dtype=torch.float32)
        local_states = torch.tensor(np.asarray(buffer.local_states, dtype=np.float32), dtype=torch.float32)
        ego_features = torch.tensor(np.asarray(buffer.ego_features, dtype=np.float32), dtype=torch.float32)
        # Actor input, same layout as in select_action(): [local, global, ego].
        states = torch.cat([local_states, global_states, ego_features], dim=1)
        actions = torch.tensor(np.asarray(buffer.actions, dtype=np.float32), dtype=torch.float32)
        logprobs = torch.tensor(np.asarray(buffer.logprobs, dtype=np.float32), dtype=torch.float32)


        # Diagnostics, averaged over all minibatches at the end.
        actor_losses = []
        critic_losses = []
        approx_kls = []
        clip_fractions = []
        entropies = []
        explained_variances = []

        # --- optimization epochs ----------------------------------------------------------------
        for _ in range(self.K):
            indices = torch.randperm(buffer.n)  # new shuffle every epoch
            e = self.eps
            kls = []  # approximate KL of the minibatches of this epoch

            for batch_start in range(0, buffer.n, self.batch_size):
                batch_end = batch_start + self.batch_size
                batch = indices[batch_start : batch_end]

                # Minibatch data.
                mb_states = states[batch]
                mb_local_states = local_states[batch]
                mb_ego_features = ego_features[batch]

                mb_actions = actions[batch]

                mb_logprobs = logprobs[batch]
                mb_advantages = advantages[batch]

                mb_returns = returns[batch]

                # Current policy: log-probabilities of the stored actions, and entropy.
                new_dist = self.actor(mb_states)
                mb_new_logprobs = new_dist.log_prob(mb_actions)
                entropy = new_dist.entropy().mean()

                # Current value estimates (critic input: [local, ego]).
                mb_new_values = self.critic(torch.cat([mb_local_states, mb_ego_features], dim=1)).squeeze(-1)

                # Probability ratio pi_new / pi_old.
                r = torch.exp(mb_new_logprobs - mb_logprobs)

                # Clipped surrogate objective (negated: we minimize) with an entropy bonus.
                loss_actor = -torch.mean(torch.minimum(mb_advantages*r, torch.clamp(r, 1-e, 1+e)*mb_advantages))
                loss_actor = loss_actor - self.entropy_coef*entropy
                # Plain MSE value loss (no value clipping).
                loss_critic = torch.mean((mb_new_values - mb_returns)**2)

                # The networks share no parameters, so both losses can be backpropagated separately.
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                loss_actor.backward()
                loss_critic.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optim.step()
                self.critic_optim.step()

                # Diagnostics (no gradients needed).
                with torch.no_grad():
                    # Low-variance KL estimator: (r - 1) - log r, with log r = new_logprob - old_logprob.
                    approx_kl = ((r-1) - (mb_new_logprobs-mb_logprobs)).mean().item()
                    # Share of samples whose ratio left the clipping range.
                    clip_fraction = ((r - 1.0).abs() > e).float().mean().item()

                    kls.append(approx_kl)

                    # Explained variance of the critic on this minibatch (1 = perfect, <= 0 = useless).
                    var_returns = torch.var(mb_returns, unbiased=False)
                    var_residual = torch.var(mb_returns - mb_new_values, unbiased=False)
                    explained_variance = 1.0 - (var_residual / (var_returns + 1e-8)).item()
                    actor_losses.append(loss_actor.item())
                    critic_losses.append(loss_critic.item())
                    approx_kls.append(approx_kl)
                    clip_fractions.append(clip_fraction)
                    entropies.append(entropy.item())
                    explained_variances.append(explained_variance)

            # Early stopping: the policy has moved far enough from the data-collecting policy.
            if np.mean(kls) > self.target_kl:
                break

        # One learning-rate decay step per update.
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
        """
        Normalize observations with running statistics.

        update: whether to update the statistics with these observations
                (None -> follow `self.is_training`).
        Ego features are returned unchanged.

        Returns: (n_local, n_ego) if global_obs is None, otherwise (n_local, n_global, n_ego).
        """
        update = self.is_training if update is None else update

        if update:
            self.local_rms.update(local_obs)
        n_local = self.local_rms.normalize(local_obs)

        # Ego features are not normalized.
        n_ego = ego_features

        if global_obs is None:
            return n_local, n_ego

        if update:
            self.global_rms.update(global_obs)

        n_global = self.global_rms.normalize(global_obs)

        return n_local, n_global, n_ego

    def normalize_reward(self, reward, done=False, update=None):
        """
        Optional reward scaling: divide the reward by the running std of the discounted return
        (no mean subtraction). `done` resets the running discounted return at episode ends.
        """
        update = self.is_training if update is None else update
        # Same trick as in select_action: initialize the mean so that scalar updates work.
        if self.ret_rms.mean is None:
            self.ret_rms.mean = np.zeros_like(self._disc_return)
        if update:
            self._disc_return = self._disc_return * self.gamma + reward
            self.ret_rms.update(self._disc_return)
            if done:
                self._disc_return = 0.0

        return self.ret_rms.scale(reward)

    def get_next_value(self, local_obs, ego_features, done=False):
        """
        Critic value of the state after the last stored step, used to bootstrap GAE at the end
        of a rollout. Returns 0 for a terminal state. Statistics are NOT updated here.
        """
        if done:
            return 0.0
        n_local, n_ego = self.normalize_obs(local_obs, ego_features=ego_features, update=False)

        with torch.no_grad():
            return float(self.critic(torch.tensor(np.concatenate([n_local, n_ego]))).squeeze(-1))


class RunningMeanStd:
    """
    Online estimate of mean and variance (parallel / Chan et al. update, which merges
    the running statistics with the statistics of a new batch).
    """

    def __init__(self, eps=1e-8):
        self.mean = None
        self.var = None
        self.count = 0
        self.eps = eps  # numerical stability in the denominator

    def update(self, x):
        """
        Merge new data into the statistics. `x` is either a single sample (same shape as `mean`)
        or a batch with the batch dimension first.
        """
        x = np.asarray(x, dtype=np.float64)
        # A single sample is treated as a batch of size one.
        if self.mean is not None and x.shape == self.mean.shape:
            x = x[np.newaxis, ...]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        if batch_count == 0:
            return

        # First call: start from mean 0 / var 1.
        if self.mean is None:
            self.mean = np.zeros_like(batch_mean)
        if self.var is None:
            self.var = np.ones_like(batch_var)

        # Combine (count, mean, var) of the running statistics and of the batch.
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
        """Standardize: (x - mean) / sqrt(var + eps), returned as float32."""
        x = np.asarray(x, dtype=np.float64)
        y = (x - self.mean) / np.sqrt(self.var + self.eps)
        return y.astype(np.float32)

    def scale(self, x):
        """Divide by the standard deviation only (no mean subtraction), returned as float32."""
        y = np.asarray(x, dtype=np.float64) / np.sqrt(self.var + self.eps)
        return y.astype(np.float32)

    def state_dict(self):
        """Statistics for saving into a checkpoint."""
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        """Restore statistics saved by state_dict()."""
        self.mean = d["mean"].copy()
        self.var = d["var"].copy()
        self.count = d["count"]