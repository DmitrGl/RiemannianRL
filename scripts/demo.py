"""
Interactive sandbox for a trained glider agent.

Loads a checkpoint from `models/` and plays episodes forever in a live tkinter window
(top view of the surface, glider in cyan, target in orange).

Run from the project root:  python -m scripts.demo

Keys: [space] pause   [r] new episode   [d] deterministic/stochastic policy   [+/-] speed   [q] quit
"""
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

WIDTH = 700       # canvas size in pixels
HEIGHT = 700
FRAME_MS = 33     # delay between frames (~30 FPS)
TRAIL_MAX = 3000  # maximum number of points kept in each trajectory trail


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
        # Half-size of the visible region (the same limit that ends an episode as "too far").
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

        # Normalization statistics saved at the end of training (frozen: only normalize() is used).
        self.local_rms = ppo.RunningMeanStd()
        self.local_rms.load_state_dict(ckpt["local_rms"])
        self.global_rms = ppo.RunningMeanStd()
        self.global_rms.load_state_dict(ckpt["global_rms"])

        # ---------------- sandbox state ----------------
        self.paused = False
        self.deterministic = True     # mean action by default; 'd' toggles sampling
        self.steps_per_frame = 3      # simulation speed
        self.episodes = 0
        self.successes = 0
        self.recent_rewards = deque(maxlen=5)  # rewards of the last 5 finished episodes
        self.episode_reward = 0.0
        self.last_reason = "-"
        # Trajectory trails in R^3 (only x, y are drawn).
        self.g_trail = [self.obs["global_state"][:3].copy()]
        self.t_trail = [self.obs["global_state"][6:9].copy()]

        # ---------------- tkinter ----------------
        self.root = tk.Tk()
        self.root.title("Geodesic glider sandbox")
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()
        self.canvas.focus_set()  # the canvas must have focus to receive key presses
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

        # Surface height on a WIDTH x WIDTH grid covering [-limit, limit]^2.
        # y runs downwards so that image row 0 is the top of the picture.
        S = WIDTH
        xs = np.linspace(-self.limit, self.limit, S)
        ys = np.linspace(self.limit, -self.limit, S)
        X, Y = np.meshgrid(xs, ys)
        Z = np.asarray(f_func(X, Y), dtype=float)

        zmin, zmax = float(Z.min()), float(Z.max())
        Zn = (Z - zmin) / max(zmax - zmin, 1e-8)  # height normalized to [0, 1]

        # Hill shading: surface normal from the height gradient (per pixel).
        # Rows go in the -y direction, hence the signs.
        dZr, dZc = np.gradient(Z)
        nx, ny, nz = -dZc, dZr, np.ones_like(Z)
        nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nlen, ny / nlen, nz / nlen
        # Light direction (normalized): from the upper left, fairly high.
        Lx, Ly, Lz = -0.45, 0.60, 0.66
        Ln = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
        Lx, Ly, Lz = Lx / Ln, Ly / Ln, Lz / Ln
        shade = np.clip(nx * Lx + ny * Ly + nz * Lz, 0.0, 1.0)  # diffuse lighting
        light = 0.30 + 0.70 * shade                              # plus ambient light

        # Color ramp: dark blue (low) to lighter blue (high), multiplied by the lighting.
        R = ((8 + 70 * Zn) * light).astype(np.uint8)
        G = ((14 + 110 * Zn) * light).astype(np.uint8)
        B = ((30 + 150 * Zn) * light).astype(np.uint8)
        rgb = np.stack([R, G, B], axis=-1)

        # Unit grid: a faint line at every integer coordinate, painted into the image
        # (brightened to at least the given level, never darkened).
        px_per_unit = S / (2 * self.limit)
        center = S // 2
        for k in range(-int(self.limit), int(self.limit) + 1):
            p = int(round(center + k * px_per_unit))
            if 0 <= p < S:
                for ch, gc in ((0, 34), (1, 46), (2, 70)):
                    rgb[p, :, ch] = np.maximum(rgb[p, :, ch], gc)
                    rgb[:, p, ch] = np.maximum(rgb[:, p, ch], gc)

        # Put the image on the canvas. Tk does not keep its own reference to the image,
        # so it is stored in self._bg to prevent garbage collection.
        self._pil = Image.fromarray(rgb, "RGB")
        img = ImageTk.PhotoImage(self._pil, master=self.root)
        self._bg = img
        self.canvas.create_image(0, 0, anchor="nw", image=img, tags="bg")

        # Contour lines: matplotlib is only used to compute the line segments;
        # they are drawn as smooth vector lines on the canvas.
        fig, ax = plt.subplots(figsize=(1, 1))
        levels = zmin + np.linspace(0.15, 0.9, 6) * (zmax - zmin)
        cs = ax.contour(X, Y, Z, levels=levels, linewidths=1.0)

        # Collect all contour polylines (different matplotlib versions expose them differently).
        paths = []
        if hasattr(cs, "allsegs"):
            for segs in cs.allsegs:
                paths.extend(segs)
        else:
            paths = [p.vertices for p in cs.get_paths()]
        for seg in paths:
            if len(seg) < 2:
                continue
            # World coordinates -> pixels (y is flipped).
            flat = []
            for xw, yw in seg:
                flat.extend((
                    (xw + self.limit) / (2 * self.limit) * WIDTH,
                    HEIGHT - (yw + self.limit) / (2 * self.limit) * HEIGHT,
                ))
            self.canvas.create_line(*flat, fill="#8fc1ff", width=1,
                                    smooth=True, tags="bg")
        plt.close(fig)

        # Tick labels along both axes (the origin is skipped).
        for k in range(-int(self.limit), int(self.limit) + 1):
            if k == 0:
                continue
            px = (k + self.limit) / (2 * self.limit) * WIDTH
            py = HEIGHT - (k + self.limit) / (2 * self.limit) * HEIGHT
            self.canvas.create_text(px + 3, center + 3, anchor="nw", fill="#5a6c8c",
                                    text=str(k), tags="bg")
            self.canvas.create_text(center + 3, py + 3, anchor="nw", fill="#5a6c8c",
                                    text=str(k), tags="bg")

        # Keep everything tagged "bg" below the trajectories drawn later.
        self.canvas.tag_lower("bg")

    # ------------------------------------------------------------------ helpers
    def to_px(self, p):
        """Convert a world point (x, y, ...) to canvas pixels (clamped to the visible region, y flipped)."""
        x = max(-self.limit, min(self.limit, float(p[0])))
        y = max(-self.limit, min(self.limit, float(p[1])))
        px = (x + self.limit) / (2 * self.limit) * WIDTH
        py = HEIGHT - (y + self.limit) / (2 * self.limit) * HEIGHT
        return px, py

    def pick_action(self):
        """Same preprocessing as in training: RMS-normalize -> actor -> tanh."""
        n_local = self.local_rms.normalize(self.obs["local_state"])
        if self.actor_obs_dim == self.critic_obs_dim:
            # Older checkpoints: the actor has the same input size as the critic
            # and sees only the normalized local state.
            obs_vec = n_local
        else:
            # Current scheme (as in PPOAgent.select_action): [local, global, ego];
            # ego features are used as is.
            n_global = self.global_rms.normalize(self.obs["global_state"])
            n_ego = self.obs["ego_features"]
            obs_vec = np.concatenate([n_local, n_global, n_ego])

        with torch.no_grad():
            d = self.actor(torch.tensor(obs_vec, dtype=torch.float32))
            # Mean of the policy (deterministic) or a sample from it (stochastic).
            action = d.mean if self.deterministic else d.sample()
        return np.tanh(action.numpy())

    # ------------------------------------------------------------------ simulation
    def step_env(self):
        """Make one environment step with the agent's action; start a new episode when it ends."""
        action = self.pick_action()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.info = info

        self.episode_reward += reward

        # Extend the trails, keeping only the last TRAIL_MAX points.
        self.g_trail.append(obs["global_state"][:3].copy())
        self.t_trail.append(obs["global_state"][6:9].copy())
        if len(self.g_trail) > TRAIL_MAX:
            self.g_trail = self.g_trail[-TRAIL_MAX:]
            self.t_trail = self.t_trail[-TRAIL_MAX:]

        # Episode finished: update the statistics and start a new one.
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
        """Start a new episode immediately ([r] key); the aborted episode is not counted."""
        self.obs, self.info = self.env.reset(options=self.opts)
        self.episode_reward = 0.0
        self.g_trail = [self.obs["global_state"][:3].copy()]
        self.t_trail = [self.obs["global_state"][6:9].copy()]

    # ------------------------------------------------------------------ main loop
    def tick(self):
        """One frame: simulate `steps_per_frame` steps (unless paused), redraw, schedule the next frame."""
        if not self.paused:
            for _ in range(self.steps_per_frame):
                self.step_env()
        self.draw()
        self.root.after(FRAME_MS, self.tick)

    def draw(self):
        """Redraw the foreground: trails, current positions and the text overlay (HUD)."""
        c = self.canvas
        c.delete("fg")  # remove the previous frame; the background ("bg") stays

        g_px = [self.to_px(p) for p in self.g_trail]
        t_px = [self.to_px(p) for p in self.t_trail]

        # Trails.
        if len(g_px) > 1:
            c.create_line(*[co for p in g_px for co in p], fill="cyan", width=1, tags="fg")
        if len(t_px) > 1:
            c.create_line(*[co for p in t_px for co in p], fill="orange", width=1, tags="fg")

        # Current positions: glider (cyan, larger) and target (orange).
        if g_px:
            x, y = g_px[-1]
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="cyan", outline="", tags="fg")
        if t_px:
            x, y = t_px[-1]
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="orange", outline="", tags="fg")

        # Text overlay.
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
        """Keyboard controls (see the module docstring)."""
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
    """Ask which saved model to load and run the sandbox."""
    if not MODELS_DIR.exists() or not list(MODELS_DIR.glob("*.pt")):
        print(f"No models found in {MODELS_DIR}. Train and save a model first.")
        return

    print("Available model versions:")
    for m in sorted(MODELS_DIR.glob("*.pt")):
        print("  -", m.stem)

    # Repeat until an existing version name is entered.
    while True:
        name = input("Version name to load: ").strip()
        path = MODELS_DIR / f"{name}.pt"
        if path.exists():
            break
        print(f"'{name}' not found, try again.")

    # weights_only=False: the checkpoint contains a config dict and numpy arrays,
    # not only tensors. Load only checkpoints you trust.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded '{name}' (step {ckpt.get('global_step')}, episodes {ckpt.get('episode_count')})")

    sandbox = Sandbox(ckpt)
    sandbox.root.mainloop()


if __name__ == "__main__":
    main()