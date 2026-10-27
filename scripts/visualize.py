from pathlib import Path
import yaml
from types import SimpleNamespace
from env.environment import GeodesicGliderEnv
import sympy as sp
import agent.network as network
import agent.ppo as ppo
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    r_manifold = sp.Matrix([u, v, sp.exp(-(u**2 + v**2)/4)])
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
    terminated = False
    truncated = False

    glider_history = []
    target_history = []

    glider_history.append(obs["global_state"][:3].copy())
    target_history.append(obs["global_state"][6:9].copy())

    max_steps = 700
    step_idx = 0

    while not (terminated or truncated) and step_idx < max_steps:
        action, logprob, state_val = agent.select_action(local_obs, global_obs)

        

        obs, reward, terminated, truncated, info = env.step(action)

        local_obs = obs["local_state"]
        global_obs = obs["global_state"]

        episode_reward += reward

        glider_history.append(global_obs[:3].copy())
        target_history.append(global_obs[6:9].copy())

        step_idx += 1

    visualize3d(glider_history, target_history, sp.lambdify([u,v],r_manifold[2]))


def visualize2d(glider_history, target_history, Z_func, save = False, show = True, save_file = "trajectory_2d"):
    glider_arr = np.asarray(glider_history, dtype=np.float32)
    target_arr = np.asarray(target_history, dtype=np.float32)

    X_g, Y_g, Z_g = glider_arr[:, 0], glider_arr[:, 1], glider_arr[:, 2]
    X_t, Y_t, Z_t = target_arr[:, 0], target_arr[:, 1], target_arr[:, 2]
    all_x = np.concatenate([X_g, X_t])
    all_y = np.concatenate([Y_g, Y_t])
    
    lim = max(
        5.0,
        float(np.abs(all_x).max()),
        float(np.abs(all_y).max())
    ) + 0.5

    grid_size = 120 if lim < 10 else 70

    x = np.linspace(-lim, lim, grid_size)
    y = np.linspace(-lim, lim, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = Z_func(X,Y)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
        
    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=20,
        cmap="viridis",
        alpha=0.55
    )
    
    fig.colorbar(contour, ax=ax)

    line_g, = ax.plot([], [], color="tab:blue", linewidth=2, label="Glider")
    line_t, = ax.plot([], [], color="tab:red", linewidth=2, linestyle="--", label="Target")
    
    dot_g, = ax.plot([], [], "o", color="tab:blue", markersize=6)
    dot_t, = ax.plot([], [], "o", color="tab:red", markersize=6)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Glider and target on manifold (2D projection)")
    
    ax.legend(loc="upper right")
    
    
    def init():
        line_g.set_data([], [])
        line_t.set_data([], [])
        dot_g.set_data([], [])
        dot_t.set_data([], [])
        return line_g, line_t, dot_g, dot_t
    
    
    def update(frame):
        i = min(frame, len(X_g) - 1)
    
        line_g.set_data(X_g[:i + 1], Y_g[:i + 1])
        line_t.set_data(X_t[:i + 1], Y_t[:i + 1])
    
        dot_g.set_data([X_g[i]], [Y_g[i]])
        dot_t.set_data([X_t[i]], [Y_t[i]])
    
        return line_g, line_t, dot_g, dot_t
    
    
    anim = FuncAnimation(
        fig,
        update,
        frames=len(X_g),
        init_func=init,
        interval=30,
        blit=True,
        repeat=True
    )
    
    if(show): plt.show()
    
    if(save):
        PATH = BASE_DIR.parent / "video" / save_file
        anim.save(PATH.with_suffix(".gif"), writer="pillow", fps=30, dpi=100)
        anim.save(PATH.with_suffix(".mp4"), writer="ffmpeg", fps=30, dpi=100)

    
def visualize3d(glider_history, target_history, Z_func, save=False, show = True, save_file = "trajectory_3d"):
    glider_arr = np.asarray(glider_history, dtype=np.float32)
    target_arr = np.asarray(target_history, dtype=np.float32)

    X_g, Y_g, Z_g = glider_arr[:, 0], glider_arr[:, 1], glider_arr[:, 2]
    X_t, Y_t, Z_t = target_arr[:, 0], target_arr[:, 1], target_arr[:, 2]
    
    all_x = np.concatenate([X_g, X_t])
    all_y = np.concatenate([Y_g, Y_t])
    
    
    lim_xy = max(
        5.0,
        float(np.abs(all_x).max()),
        float(np.abs(all_y).max())
    ) + 0.5

    
    grid_size = 60 if lim_xy < 10 else 40 

    x = np.linspace(-lim_xy, lim_xy, grid_size)
    y = np.linspace(-lim_xy, lim_xy, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = Z_func(X, Y)

    
    z_min = float(min(Z_g.min(), Z_t.min(), Z.min())) - 0.2
    z_max = float(max(Z_g.max(), Z_t.max(), Z.max())) + 0.2

    fig = plt.figure(figsize=(9, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, edgecolor='none')
    
    
    line_g, = ax.plot([], [], [], color="tab:blue", linewidth=2, label="Glider")
    line_t, = ax.plot([], [], [], color="tab:red", linewidth=2, linestyle="--", label="Target")
    
    dot_g, = ax.plot([], [], [], "o", color="tab:blue", markersize=6)
    dot_t, = ax.plot([], [], [], "o", color="tab:red", markersize=6)
    
    
    ax.set_xlim(-lim_xy, lim_xy)
    ax.set_ylim(-lim_xy, lim_xy)
    ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Glider and target on manifold (3D)")
    ax.legend(loc="upper left")
    
    
    ax.view_init(elev=30, azim=45)
    
    def init():
        line_g.set_data_3d([], [], [])
        line_t.set_data_3d([], [], [])
        dot_g.set_data_3d([], [], [])
        dot_t.set_data_3d([], [], [])
        return line_g, line_t, dot_g, dot_t
    
    def update(frame):
        i = min(frame, len(X_g) - 1)
    
        
        line_g.set_data_3d(X_g[:i + 1], Y_g[:i + 1], Z_g[:i + 1])
        line_t.set_data_3d(X_t[:i + 1], Y_t[:i + 1], Z_t[:i + 1])
    
        dot_g.set_data_3d([X_g[i]], [Y_g[i]], [Z_g[i]])
        dot_t.set_data_3d([X_t[i]], [Y_t[i]], [Z_t[i]])
    
        return line_g, line_t, dot_g, dot_t
    
    
    anim = FuncAnimation(
        fig,
        update,
        frames=len(X_g),
        init_func=init,
        interval=30,
        blit=False, 
        repeat=True
    )
    
    if(show): plt.show()
    
    if save:
        PATH = BASE_DIR.parent / "video" / save_file
        anim.save(PATH.with_suffix(".gif"), writer="pillow", fps=30, dpi=100)
        anim.save(PATH.with_suffix(".mp4"), writer="ffmpeg", fps=30, dpi=100)

if __name__ == "__main__":
    main()