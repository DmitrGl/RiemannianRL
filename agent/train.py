from pathlib import Path
import yaml
from types import SimpleNamespace
from env.environment import GeodesicGliderEnv
import sympy as sp
import network
import ppo
import torch
import numpy as np
import scripts.visualize as viz
import wandb
import math


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR.parent / "configs" / "ppo_config.yaml"

def to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    return d
    
def main():
    print("Begin training process: Initialization.")

    with open(CONFIG_PATH) as f:
        cfg_dict = yaml.safe_load(f)
        cfg = to_namespace(cfg_dict)
    
    use_wandb = cfg.logging.use_wandb
    
    if(use_wandb):
        run = wandb.init(
            project="riemannian-rl-glider",
            name="test-run",
            config=cfg_dict,
        )
    else: run = None
    
    
    u, v = sp.symbols('u v')
    r_manifold = sp.Matrix([u, v, sp.exp(-(u**2 + v**2))])
    env = GeodesicGliderEnv(r_manifold, cords= [u,v])
    local_obs_dim = env.observation_space["local_state"].shape[0]
    obs_dim = local_obs_dim + env.observation_space["global_state"].shape[0] + env.observation_space["ego_features"].shape[0]
    act_dim = env.action_space.shape[0]
    
    actor = network.Actor(obs_dim, act_dim, hidden= cfg.model.hidden_size)
    critic = network.Critic(local_obs_dim, hidden= cfg.model.hidden_size)
    buffer = ppo.RolloutBuffer(cfg.training.rollout_steps, cfg.training.gamma, cfg.training.lam)
    T = cfg.training.total_timesteps
    agent = ppo.PPOAgent(
                        actor,
                        critic,
                        cfg.actor_optimizer.lr,
                        cfg.critic_optimizer.lr, 
                        cfg.training.epochs, 
                        cfg.training.epsilon_clip, 
                        cfg.training.batch_size,
                        cfg.training.ent_coef,
                        cfg.training.target_kl, 
                        cfg.training.gamma,
                        math.floor(T/buffer.n)
                        )

    print("Initialization complete, begin training.")
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
    obs, info = env.reset(options= options_dict)
    local_obs = obs["local_state"]
    global_obs = obs["global_state"]
    ego_features = obs["ego_features"]
    next_value = -1
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    episode_initial_distance = info["distance"]
    episode_min_distance = float("inf")
    global_step = 0
    total_rewards = []
    total_success = []
    total_too_far = []
    total_timeout = []
    glider_history = []
    target_history = []
    glider_history.append(obs["global_state"][:3].copy())
    target_history.append(obs["global_state"][6:9].copy())
    until_log = cfg.logging.log_interval
    until_video = cfg.logging.video_interval
    for i in range(T):
    #while(1):
        action, logprob, state_val, n_local_obs, n_global_obs, n_ego = agent.select_action(local_obs, global_obs, ego_features)
        
        obs, reward, terminated, truncated, info = env.step(np.tanh(action))
        done = terminated or truncated
        
        n_reward = agent.normalize_reward(reward, done=done)


        buffer.store(n_local_obs, n_global_obs, n_ego, action, logprob, n_reward, state_val, (terminated or truncated))
        local_obs = obs["local_state"]
        global_obs = obs["global_state"]
        ego_features = obs["ego_features"]
        episode_reward += reward
        episode_length += 1
        episode_min_distance = min(episode_min_distance, info["distance"])
        global_step += 1
        glider_history.append(obs["global_state"][:3].copy())
        target_history.append(obs["global_state"][6:9].copy())
        
        if done:
            #end of episode
            reason = info["reason"]
            #print(f"reason: {reason}")
            
            if(use_wandb):
                wandb.log(
                    {
                        "episode/reward": episode_reward,
                        "episode/length": episode_length,
                        "episode/initial_distance": episode_initial_distance,
                        "episode/min_distance": episode_min_distance,
                    },
                    step= global_step,
                )
            
            obs, info = env.reset(options= options_dict)
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
            if until_log == 0:
                until_log = cfg.logging.log_interval
                if(use_wandb):
                    wandb.log(
                        {
                            "episode/average_reward": np.mean(total_rewards),
                            "episode/success_rate": np.mean(total_success),
                            "episode/timeout_rate": np.mean(total_timeout),
                            "episode/too_far_rate": np.mean(total_too_far),
                            "episode/length": episode_length,
                            
                        },
                        step= global_step,
                    )
                total_rewards.clear()
                total_success.clear()
                total_timeout.clear()
                total_too_far.clear()
            if until_video == 0:
                until_video = cfg.logging.video_interval
                video_file = f"traj_ep{episode_count}"
                viz.visualize2d(
                    glider_history, 
                    target_history, 
                    sp.lambdify([u,v], r_manifold[2]),
                    show=False,          
                    save=True,
                    save_file=video_file
                )
                if(use_wandb):
                    PATH = BASE_DIR.parent / "video" / video_file
                    wandb.log(
                        {
                            "video/trajectory_2d": wandb.Video(PATH.with_suffix(".mp4"), fps=30, format="mp4"),
                            "episode/reward": episode_reward,  
                            "episode/reason": reason,
                        }, 
                        step=global_step
                    )
            episode_length = 0
            episode_reward = 0
            glider_history = []
            target_history = []
            glider_history.append(obs["global_state"][:3].copy())
            target_history.append(obs["global_state"][6:9].copy())


        if(buffer.cur == buffer.n):
            #end of rollout
            next_value = agent.get_next_value(local_obs, done=done)
            
            metrics = agent.update(buffer, next_value)
            buffer.clear()

            if(use_wandb):
                wandb_metrics = {}

                for key, value in metrics.items():
                    wandb_metrics[f"train/{key}"] = value
                wandb.log(
                    wandb_metrics,
                    step = global_step
                )
            
    wandb.finish()
    print("End of trainig process: Training complete")



if __name__ == "__main__":
    main()