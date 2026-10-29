from collections import deque

from pathlib import Path
import tkinter as tk

import numpy as np
import torch
import sympy as sp

from env.environment import GeodesicGliderEnv
import agent.network as network
import agent.ppo as ppo

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

WIDTH = 700
HEIGHT = 700
FRAME_MS = 33          
TRAIL_MAX = 3000


class Sandbox:
    """
    Infinite playground: loads a trained checkpoint and plays episodes forever
    in a live tkinter window.
    """

    def __init__(self, ckpt):
        self.ckpt = ckpt
        cfg = ckpt["config"]
        self.opts = ckpt["options"]

        # ---------------- environment (same manifold as during training) ----------------
        u, v = sp.symbols("u v")
        f_expr = sp.sympify(ckpt["manifold_expr"])
        r_manifold = sp.Matrix([u, v, f_expr])
        self.env = GeodesicGliderEnv(r_manifold, cords=[u, v])
        self.obs, self.info = self.env.reset(options=self.opts)
        self.limit = float(self.opts.get("cord_limit", 5.0))

        # ---------------- agent ----------------
        self.actor_obs_dim = ckpt["actor_obs_dim"]
        self.critic_obs_dim = ckpt["critic_obs_dim"]

        self.actor = network.Actor(
            self.actor_obs_dim, ckpt["act_dim"], hidden=cfg["model"]["hidden_size"]
        )
        self.critic = network.Critic(self.critic_obs_dim, hidden=cfg["model"]["hidden_size"])
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor.eval()
        self.critic.eval()

        self.local_rms = ppo.RunningMeanStd()
        self.local_rms.load_state_dict(ckpt["local_rms"])
        self.global_rms = ppo.RunningMeanStd()
        self.global_rms.load_state_dict(ckpt["global_rms"])
        self.ego_rms = ppo.RunningMeanStd()
        self.ego_rms.load_state_dict(ckpt["ego_rms"])

        # ---------------- sandbox state ----------------
        self.paused = False
        self.deterministic = True     # mean action by default; 'd' toggles sampling
        self.steps_per_frame = 3      # simulation speed
        self.episodes = 0
        self.successes = 0
        self.recent_rewards = deque(maxlen=5)
        self.episode_reward = 0.0
        self.last_reason = "-"
        self.g_trail = [self.obs["global_state"][:3].copy()]
        self.t_trail = [self.obs["global_state"][6:9].copy()]

        # ---------------- tkinter ----------------
        self.root = tk.Tk()
        self.root.title("Geodesic glider sandbox")
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()
        self.canvas.focus_set()
        self.canvas.bind("<Key>", self.on_key)

        self._make_background(f_expr)
        self.root.after(FRAME_MS, self.tick)

    # ------------------------------------------------------------------ background
    def _make_background(self, f_expr):
        """Hill-shaded heightmap + smooth vector contours + unit grid + tick labels."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image, ImageTk

        u, v = sp.symbols("u v")
        f_func = sp.lambdify((u, v), f_expr)

        S = WIDTH
        xs = np.linspace(-self.limit, self.limit, S)
        ys = np.linspace(self.limit, -self.limit, S)      
        X, Y = np.meshgrid(xs, ys)
        Z = np.asarray(f_func(X, Y), dtype=float)

        zmin, zmax = float(Z.min()), float(Z.max())
        Zn = (Z - zmin) / max(zmax - zmin, 1e-8)

        
        dZr, dZc = np.gradient(Z)                          
        nx, ny, nz = -dZc, dZr, np.ones_like(Z)            
        nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nlen, ny / nlen, nz / nlen
        Lx, Ly, Lz = -0.45, 0.60, 0.66                     
        Ln = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
        Lx, Ly, Lz = Lx / Ln, Ly / Ln, Lz / Ln
        shade = np.clip(nx * Lx + ny * Ly + nz * Lz, 0.0, 1.0)
        light = 0.30 + 0.70 * shade                        

        
        R = ((8 + 70 * Zn) * light).astype(np.uint8)
        G = ((14 + 110 * Zn) * light).astype(np.uint8)
        B = ((30 + 150 * Zn) * light).astype(np.uint8)
        rgb = np.stack([R, G, B], axis=-1)

        
        px_per_unit = S / (2 * self.limit)
        center = S // 2
        for k in range(-int(self.limit), int(self.limit) + 1):
            p = int(round(center + k * px_per_unit))
            if 0 <= p < S:
                for ch, gc in ((0, 34), (1, 46), (2, 70)):
                    rgb[p, :, ch] = np.maximum(rgb[p, :, ch], gc)
                    rgb[:, p, ch] = np.maximum(rgb[:, p, ch], gc)

        self._pil = Image.fromarray(rgb, "RGB")            
        img = ImageTk.PhotoImage(self._pil, master=self.root)
        self._bg = img
        self.canvas.create_image(0, 0, anchor="nw", image=img, tags="bg")

        
        fig, ax = plt.subplots(figsize=(1, 1))
        levels = zmin + np.linspace(0.15, 0.9, 6) * (zmax - zmin)
        cs = ax.contour(X, Y, Z, levels=levels, linewidths=1.0)

        paths = []
        if hasattr(cs, "allsegs"):                         
            for segs in cs.allsegs:
                paths.extend(segs)
        else:                                              
            paths = [p.vertices for p in cs.get_paths()]
        for seg in paths:
            if len(seg) < 2:
                continue
            flat = []
            for xw, yw in seg:
                flat.extend((
                    (xw + self.limit) / (2 * self.limit) * WIDTH,
                    HEIGHT - (yw + self.limit) / (2 * self.limit) * HEIGHT,
                ))
            self.canvas.create_line(*flat, fill="#8fc1ff", width=1,
                                    smooth=True, tags="bg")
        plt.close(fig)

        
        for k in range(-int(self.limit), int(self.limit) + 1):
            if k == 0:
                continue
            px = (k + self.limit) / (2 * self.limit) * WIDTH
            py = HEIGHT - (k + self.limit) / (2 * self.limit) * HEIGHT
            self.canvas.create_text(px + 3, center + 3, anchor="nw", fill="#5a6c8c",
                                    text=str(k), tags="bg")
            self.canvas.create_text(center + 3, py + 3, anchor="nw", fill="#5a6c8c",
                                    text=str(k), tags="bg")

        self.canvas.tag_lower("bg")
    
    # ------------------------------------------------------------------ helpers
    def to_px(self, p):
        x = max(-self.limit, min(self.limit, float(p[0])))
        y = max(-self.limit, min(self.limit, float(p[1])))
        px = (x + self.limit) / (2 * self.limit) * WIDTH
        py = HEIGHT - (y + self.limit) / (2 * self.limit) * HEIGHT
        return px, py

    def pick_action(self):
        """Same preprocessing as in training: RMS-normalize -> actor -> tanh."""
        n_local = self.local_rms.normalize(self.obs["local_state"])
        if self.actor_obs_dim == self.critic_obs_dim:
            obs_vec = n_local                                   # ego-only actor
        else:
            n_global = self.global_rms.normalize(self.obs["global_state"])
            n_ego = self.ego_rms.normalize(self.obs["ego_features"])
            obs_vec = np.concatenate([n_local, n_global, n_ego])       # old scheme

        with torch.no_grad():
            d = self.actor(torch.tensor(obs_vec, dtype=torch.float32))
            action = d.mean if self.deterministic else d.sample()
        return np.tanh(action.numpy())

    # ------------------------------------------------------------------ simulation
    def step_env(self):
        action = self.pick_action()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.info = info

        self.episode_reward += reward          


        self.g_trail.append(obs["global_state"][:3].copy())
        self.t_trail.append(obs["global_state"][6:9].copy())
        if len(self.g_trail) > TRAIL_MAX:
            self.g_trail = self.g_trail[-TRAIL_MAX:]
            self.t_trail = self.t_trail[-TRAIL_MAX:]

        if terminated or truncated:
            self.episodes += 1
            self.recent_rewards.append(self.episode_reward)   
            self.episode_reward = 0.0                         

            self.last_reason = info["reason"]
            if info["reason"] == "completed":
                self.successes += 1
            self.obs, self.info = self.env.reset(options=self.opts)
            self.g_trail = [self.obs["global_state"][:3].copy()]
            self.t_trail = [self.obs["global_state"][6:9].copy()]

    def force_reset(self):
        self.obs, self.info = self.env.reset(options=self.opts)
        self.episode_reward = 0.0
        self.g_trail = [self.obs["global_state"][:3].copy()]
        self.t_trail = [self.obs["global_state"][6:9].copy()]

    # ------------------------------------------------------------------ main loop
    def tick(self):
        if not self.paused:
            for _ in range(self.steps_per_frame):
                self.step_env()
        self.draw()
        self.root.after(FRAME_MS, self.tick)

    def draw(self):
        c = self.canvas
        c.delete("fg")

        g_px = [self.to_px(p) for p in self.g_trail]
        t_px = [self.to_px(p) for p in self.t_trail]

        if len(g_px) > 1:
            c.create_line(*[co for p in g_px for co in p], fill="cyan", width=1, tags="fg")
        if len(t_px) > 1:
            c.create_line(*[co for p in t_px for co in p], fill="orange", width=1, tags="fg")

        if g_px:
            x, y = g_px[-1]
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="cyan", outline="", tags="fg")
        if t_px:
            x, y = t_px[-1]
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="orange", outline="", tags="fg")

        dist = self.info.get("distance", float("nan"))
        
        rew_str = ", ".join(f"{r:.1f}" for r in self.recent_rewards) if self.recent_rewards else "-"
        mean5 = float(np.mean(self.recent_rewards)) if self.recent_rewards else float("nan")

        hud = (
            f"episodes: {self.episodes}   success: {self.successes}   last: {self.last_reason}\n"
            f"current episode reward: {self.episode_reward:+.2f}\n"
            f"last-5 rewards: [{rew_str}]   mean5: {mean5:.1f}\n"
            f"time: {self.env.time:.2f}/{self.env.timeout:.0f}   dist: {dist:.3f}\n"
            f"mode: {'deterministic' if self.deterministic else 'stochastic'}   "
            f"speed: x{self.steps_per_frame}   {'PAUSED' if self.paused else 'running'}\n"
            f"[space] pause   [r] reset   [d] det/stoch   [+/-] speed   [q] quit"
        )
        c.create_text(8, 8, anchor="nw", fill="white", justify="left", text=hud, tags="fg")

    # ------------------------------------------------------------------ controls
    def on_key(self, event):
        k = event.keysym
        if k == "space":
            self.paused = not self.paused
        elif k == "r":
            self.force_reset()
        elif k == "d":
            self.deterministic = not self.deterministic
        elif k in ("plus", "equal"):
            self.steps_per_frame = min(50, self.steps_per_frame + 1)
        elif k in ("minus", "underscore"):
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
        elif k in ("q", "Escape"):
            self.root.destroy()


def main():
    if not MODELS_DIR.exists() or not list(MODELS_DIR.glob("*.pt")):
        print(f"No models found in {MODELS_DIR}. Train and save a model first.")
        return

    print("Available model versions:")
    for m in sorted(MODELS_DIR.glob("*.pt")):
        print("  -", m.stem)

    while True:
        name = input("Version name to load: ").strip()
        path = MODELS_DIR / f"{name}.pt"
        if path.exists():
            break
        print(f"'{name}' not found, try again.")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded '{name}' (step {ckpt.get('global_step')}, episodes {ckpt.get('episode_count')})")

    sandbox = Sandbox(ckpt)
    sandbox.root.mainloop()


if __name__ == "__main__":
    main()