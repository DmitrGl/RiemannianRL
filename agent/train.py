"""
Training script: PPO agent vs. GeodesicGliderEnv.

Pipeline: load config -> build environment (surface) and networks -> collect rollouts and update
the agent every `rollout_steps` steps -> log to Weights & Biases -> optionally save a checkpoint.

Run from the project root:  python -m agent.train
"""
from pathlib import Path
import yaml
from types import SimpleNamespace
from env.environment import GeodesicGliderEnv
import sympy as sp
import agent.network as network
import agent.ppo as ppo
import torch
import numpy as np
import scripts.visualize as viz
import wandb
import math


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR.parent / "configs" / "ppo_config.yaml"


def to_namespace(d):
    """Recursively convert nested dicts into SimpleNamespace objects (cfg["a"]["b"] -> cfg.a.b)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    return d


def main():
    print("Begin training process: Initialization.")

    # ------------------------------------------------------------------ config and logging
    with open(CONFIG_PATH) as f:
        cfg_dict = yaml.safe_load(f)  # plain dict (saved into the checkpoint)
        cfg = to_namespace(cfg_dict)  # attribute-style access for convenience

    use_wandb = cfg.logging.use_wandb

    if use_wandb:
        run = wandb.init(
            project="riemannian-rl-glider",
            name="test-run",
            config=cfg_dict,
        )
    else:
        run = None

    # ------------------------------------------------------------------ environment
    # The surface is r(u, v) = (u, v, f(u, v)).
    u, v = sp.symbols('u v')
    # Alternative surfaces to try:
    # r_manifold = sp.Matrix([u, v, sp.exp(-(u**2 + v**2))])
    # r_manifold = sp.Matrix([u, v, u+v])
    # Current surface: three Gaussian bumps/wells plus a small sinusoidal ripple.
    f = (
        1.1 * sp.exp(-((u - 2.0)**2 + (v - 1.0)**2) / 2.0)
        - 1.2 * sp.exp(-((u + 2.0)**2 + (v + 2.0)**2) / 2.5)
        + 0.7 * sp.exp(-((u + 1.0)**2 + (v - 2.0)**2) / 4.0)
        + 0.25 * sp.sin(0.9 * u) * sp.cos(0.9 * v)
    )
    r_manifold = sp.Matrix([u, v, f])

    env = GeodesicGliderEnv(r_manifold, cords=[u, v])

    # ------------------------------------------------------------------ networks and agent
    # Observation sizes: actor sees [local, global, ego], critic sees [local, ego].
    local_obs_dim = env.observation_space["local_state"].shape[0]
    obs_dim = (
        local_obs_dim
        + env.observation_space["global_state"].shape[0]
        + env.observation_space["ego_features"].shape[0]
    )
    critic_dim = local_obs_dim + env.observation_space["ego_features"].shape[0]
    actor_dim = obs_dim
    act_dim = env.action_space.shape[0]

    actor = network.Actor(actor_dim, act_dim, hidden=cfg.model.hidden_size)
    critic = network.Critic(critic_dim, hidden=cfg.model.hidden_size)
    buffer = ppo.RolloutBuffer(cfg.training.rollout_steps, cfg.training.gamma, cfg.training.lam)
    T = cfg.training.total_timesteps  # total number of environment steps
    # Arguments are positional: they must stay in the order of PPOAgent.__init__.
    agent = ppo.PPOAgent(
        actor=actor,
        critic=critic,
        lr_actor=cfg.actor_optimizer.lr,
        lr_critic=cfg.critic_optimizer.lr,
        K_epochs=cfg.training.epochs,
        eps_clip=cfg.training.epsilon_clip,
        batch_size=cfg.training.batch_size,
        entropy_coef=cfg.training.ent_coef,
        target_kl=cfg.training.target_kl,
        gamma=cfg.training.gamma,
        total_updates=math.floor(T/buffer.n) # total_updates: number of PPO updates (for the lr decay)
    )

    print("Initialization complete, begin training.")

    # Environment options, passed to every env.reset().
    options_dict = {
        'timeout': cfg.environment.timeout,
        'cord_limit': cfg.environment.cord_limit,
        'maximum_thrust': cfg.environment.maximum_thrust,
        'complete_region': cfg.environment.complete_region,
        'distance_reward_scale': cfg.environment.distance_reward_scale,
        'mu_gl': cfg.environment.mu_gl,
        'sigma_gl': cfg.environment.sigma_gl,
        'mu_tg': cfg.environment.mu_tg,
        'sigma_tg': cfg.environment.sigma_tg,
        'mu_gl_vel': cfg.environment.mu_gl_vel,
        'sigma_gl_vel': cfg.environment.sigma_gl_vel,
        'mu_tg_vel': cfg.environment.mu_tg_vel,
        'sigma_tg_vel': cfg.environment.sigma_tg_vel,
        'N': cfg.environment.N
    }

    # ------------------------------------------------------------------ training state
    obs, info = env.reset(options=options_dict)
    local_obs = obs["local_state"]
    global_obs = obs["global_state"]
    ego_features = obs["ego_features"]
    next_value = -1

    # Per-episode statistics.
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    episode_initial_distance = info["distance"]
    episode_min_distance = float("inf")
    global_step = 0

    # Statistics accumulated between two aggregated logs (cleared after each log).
    total_rewards = []
    total_success = []
    total_too_far = []
    total_timeout = []

    # Trajectory of the current episode (for the video), in R^3.
    glider_history = []
    target_history = []
    glider_history.append(obs["global_state"][:3].copy())
    target_history.append(obs["global_state"][6:9].copy())

    # Episodes left until the next aggregated log / the next video.
    until_log = cfg.logging.log_interval
    until_video = cfg.logging.video_interval

    # ------------------------------------------------------------------ main loop
    for i in range(T):  # one iteration = one environment step
        # Sample a raw action; the returned observations are normalized and go into the buffer.
        action, logprob, state_val, n_local_obs, n_global_obs, n_ego = agent.select_action(local_obs, global_obs, ego_features)

        # tanh squashing is applied here, outside the policy: PPO works with the raw action
        # and its log-probability, the environment receives tanh(action) in [-1, 1].
        obs, reward, terminated, truncated, info = env.step(np.tanh(action))
        done = terminated or truncated

        # Reward normalization is disabled (see agent.normalize_reward).
        n_reward = reward

        
        buffer.store(n_local_obs, n_global_obs, n_ego, action, logprob, n_reward, state_val, done)
        local_obs = obs["local_state"]
        global_obs = obs["global_state"]
        ego_features = obs["ego_features"]
        episode_reward += reward
        episode_length += 1
        episode_min_distance = min(episode_min_distance, info["distance"])
        global_step += 1
        glider_history.append(obs["global_state"][:3].copy())
        target_history.append(obs["global_state"][6:9].copy())

        # ---------------------------------------------------------- PPO update (end of rollout)
        if buffer.cur == buffer.n:
            # Bootstrap value of the state after the last step (0 if the episode just ended).
            next_value = agent.get_next_value(local_obs, ego_features, done=done)

            metrics = agent.update(buffer, next_value)
            buffer.clear()

            if use_wandb:
                wandb_metrics = {}

                for key, value in metrics.items():
                    wandb_metrics[f"train/{key}"] = value
                wandb.log(
                    wandb_metrics,
                    step=global_step
                )

        # ---------------------------------------------------------- end of episode
        if done:
            reason = info["reason"]  # "completed" / "too far" / "timeout"

            # Per-episode metrics.
            if use_wandb:
                wandb.log(
                    {
                        "episode/reward": episode_reward,
                        "episode/length": episode_length,
                        "episode/initial_distance": episode_initial_distance,
                        "episode/min_distance": episode_min_distance,
                    },
                    step=global_step,
                )

            # Start a new episode.
            obs, info = env.reset(options=options_dict)
            local_obs = obs["local_state"]
            global_obs = obs["global_state"]
            ego_features = obs["ego_features"]
            total_rewards.append(episode_reward)
            total_success.append(1 if reason == 'completed' else 0)
            total_too_far.append(1 if reason == 'too far' else 0)
            total_timeout.append(1 if reason == 'timeout' else 0)
            episode_initial_distance = info["distance"]
            episode_min_distance = float("inf")

            episode_count += 1
            until_log -= 1
            until_video -= 1

            # Aggregated metrics over the last `log_interval` episodes.
            if until_log == 0:
                until_log = cfg.logging.log_interval
                if use_wandb:
                    wandb.log(
                        {
                            "episode/average_reward": np.mean(total_rewards),
                            "episode/success_rate": np.mean(total_success),
                            "episode/timeout_rate": np.mean(total_timeout),
                            "episode/too_far_rate": np.mean(total_too_far),
                            "episode/length": episode_length,  # length of the last episode

                        },
                        step=global_step,
                    )
                total_rewards.clear()
                total_success.clear()
                total_timeout.clear()
                total_too_far.clear()

            # Render the trajectory of the episode that just finished.
            if until_video == 0:
                until_video = cfg.logging.video_interval
                video_file = f"traj_ep{episode_count}"
                # Saves video/<video_file>.gif and video/<video_file>.mp4.
                viz.visualize3d(
                    glider_history,
                    target_history,
                    sp.lambdify([u, v], r_manifold[2]),
                    show=False,
                    rotate=True,
                    save=True,
                    save_file=video_file
                )
                if use_wandb:
                    PATH = BASE_DIR.parent / "video" / video_file
                    wandb.log(
                        {
                            "video/trajectory_3d": wandb.Video(PATH.with_suffix(".mp4"), fps=30, format="mp4"),
                            "episode/reward": episode_reward,
                            "episode/reason": reason,
                        },
                        step=global_step
                    )

            # Reset per-episode counters and start recording the new trajectory.
            episode_length = 0
            episode_reward = 0
            glider_history = []
            target_history = []
            glider_history.append(obs["global_state"][:3].copy())
            target_history.append(obs["global_state"][6:9].copy())

    # ------------------------------------------------------------------ finish
    wandb.finish()
    print("End of training process: Training complete")

    # ================= save trained model under a version name =================
    models_dir = BASE_DIR.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ask for a version name until the checkpoint is saved or saving is skipped (empty input).
    while True:
        version = input("\nEnter model version name to save (empty = skip): ").strip()
        if version == "":
            print("Saving skipped.")
            break

        save_path = models_dir / f"{version}.pt"
        if save_path.exists():
            # Confirm before overwriting an existing version; anything but "y" asks for a new name.
            answer = input(f"Version '{version}' already exists. Overwrite? [Y/N]: ").strip().lower()
            if answer != "y" and answer != "Y":
                continue

        torch.save(
            {
                # weights
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                # observation normalization statistics (important for inference!)
                "local_rms": agent.local_rms.state_dict(),
                "global_rms": agent.global_rms.state_dict(),

                # everything needed to rebuild the exact same env + networks
                "config": cfg_dict,
                "options": options_dict,
                "manifold_expr": str(r_manifold[2]),
                "actor_obs_dim": actor_dim,
                "critic_obs_dim": critic_dim,
                "act_dim": act_dim,
                # metadata
                "global_step": global_step,
                "episode_count": episode_count,
            },
            save_path,
        )
        print(f"Model saved: {save_path}")
        break


if __name__ == "__main__":
    main()