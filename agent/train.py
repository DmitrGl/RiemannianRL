from pathlib import Path
import yaml
from types import SimpleNamespace
from env.environment import GeodesicGliderEnv
import sympy as sp
import network
import ppo
import torch
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR.parent / "configs" / "ppo_config.yaml"

def to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    return d
    
def main():
    print("Begin training process: Initialization.")

    with open(CONFIG_PATH) as f:
        cfg = to_namespace(yaml.safe_load(f))
    
    u, v = sp.symbols('u v')
    r_manifold = sp.Matrix([u, v, sp.exp(-(u**2 + v**2))])
    env = GeodesicGliderEnv(r_manifold, cords= [u,v])
    local_obs_dim = env.observation_space["local_state"].shape[0]
    obs_dim = local_obs_dim + env.observation_space["global_state"].shape[0]
    act_dim = env.action_space.shape[0]
    
    actor = network.Actor(obs_dim, act_dim, hidden= cfg.model.hidden_size)
    critic = network.Critic(local_obs_dim, hidden= cfg.model.hidden_size)
    buffer = ppo.RolloutBuffer(cfg.training.rollout_steps, cfg.training.gamma, cfg.training.lam)
    agent = ppo.PPOAgent(actor,critic,cfg.actor_optimizer.lr, cfg.critic_optimizer.lr, cfg.training.epochs, cfg.training.epsilon_clip)

    print("Initialization complete, begin training.")
    T = cfg.training.total_timesteps
    options_dict = {"mu_tg" : 0,
                    "sigma_tg": 0,
                    "mu_gl": 1,
                    "sigma_gl": 0.3
                    }
    obs, info = env.reset(options= options_dict)
    local_obs = obs["local_state"]
    global_obs = obs["global_state"]
    next_value = -1
    episode_reward = 0
    total_rewards = []
    until_log = cfg.logging.log_interval
    for i in range(T):
        action, logprob, state_val = agent.select_action(local_obs, global_obs)
        obs, reward, terminated, truncated, info = env.step(action)
        buffer.store(local_obs, global_obs, action, logprob, reward, state_val, (terminated or truncated))
        local_obs = obs["local_state"]
        global_obs = obs["global_state"]
        episode_reward += reward
        
        if terminated or truncated:
            #end of episode
            next_value = 0
            reason = info["reason"]
            #print("="*50)
            #print("End of episode:")
            #print(f"local_obs: {local_obs}")
            #print(f"global_obs: {global_obs}")
            print(f"reason: {reason}")
            #print("="*50)
            obs, info = env.reset(options= options_dict)
            local_obs = obs["local_state"]
            global_obs = obs["global_state"]
            total_rewards.append(episode_reward)
            episode_reward = 0
            until_log -= 1
            if until_log == 0:
                until_log = cfg.logging.log_interval
                print(f"Logging: average reward within {until_log} episodes ----- {np.mean(total_rewards)}\n")
                total_rewards.clear()


        if(buffer.cur == buffer.n):
            #end of rollout 
            if next_value == -1:
                next_value = critic(torch.tensor(local_obs, dtype = torch.float32)).item()
            
            agent.update(buffer, next_value)
            buffer.clear()
            
            next_value = -1
    print("End of trainig process: Training complete")



if __name__ == "__main__":
    main()