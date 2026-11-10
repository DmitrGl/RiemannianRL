"""
Trajectory visualization: animated 2D (top view) and 3D plots of the glider and the target.

Used by agent/train.py to render videos of finished episodes.
Both functions take the recorded trajectories (lists of points in R^3) and a function Z_func(X, Y)
for the surface height z = f(u, v), and can save the animation as .gif and .mp4
(requires ffmpeg for .mp4).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: figures are rendered to files only
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

BASE_DIR = Path(__file__).resolve().parent


def visualize2d(glider_history, target_history, Z_func, save=False, show=True, save_file="trajectory_2d"):
    """
    Animate the trajectories in the (x, y) plane over a filled contour plot of the surface height.

    glider_history, target_history: sequences of points (x, y, z) in R^3, one per step
    Z_func: vectorized function Z_func(X, Y) -> height of the surface
    save: write video/<save_file>.gif and .mp4 (the `video` folder must exist; `save_file`
          should not contain dots, they are treated as a file extension)
    show: call plt.show()
    """
    glider_arr = np.asarray(glider_history, dtype=np.float32)
    target_arr = np.asarray(target_history, dtype=np.float32)

    X_g, Y_g, Z_g = glider_arr[:, 0], glider_arr[:, 1], glider_arr[:, 2]
    X_t, Y_t, Z_t = target_arr[:, 0], target_arr[:, 1], target_arr[:, 2]
    all_x = np.concatenate([X_g, X_t])
    all_y = np.concatenate([Y_g, Y_t])

    # Plot limits: at least [-5.5, 5.5], expanded to fit both trajectories.
    lim = max(
        5.0,
        float(np.abs(all_x).max()),
        float(np.abs(all_y).max())
    ) + 0.5

    # Coarser grid for large areas to keep rendering fast.
    grid_size = 120 if lim < 10 else 70

    x = np.linspace(-lim, lim, grid_size)
    y = np.linspace(-lim, lim, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = Z_func(X, Y)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

    # Surface height as a semi-transparent filled contour plot.
    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=20,
        cmap="viridis",
        alpha=0.55
    )

    fig.colorbar(contour, ax=ax)

    # Animated artists: trails (lines) and current positions (dots).
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
        """Start of the animation: empty trails and no dots."""
        line_g.set_data([], [])
        line_t.set_data([], [])
        dot_g.set_data([], [])
        dot_t.set_data([], [])
        return line_g, line_t, dot_g, dot_t

    def update(frame):
        """Draw the trajectories up to step `frame`."""
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

    # NOTE: with the "Agg" backend selected above, plt.show() does not open a window.
    if show:
        plt.show()

    if save:
        PATH = BASE_DIR.parent / "video" / save_file
        anim.save(PATH.with_suffix(".gif"), writer="pillow", fps=30, dpi=100)
        anim.save(PATH.with_suffix(".mp4"), writer="ffmpeg", fps=30, dpi=100)


def visualize3d(glider_history, target_history, Z_func,
                save=False, show=True, save_file="trajectory_3d",
                rotate=True, n_rotations=1,
                show_progress=True, elev=30):
    """
    Animate the trajectories on the 3D surface.

    glider_history, target_history: sequences of points (x, y, z) in R^3, one per step
    Z_func: vectorized function Z_func(X, Y) -> height of the surface
    save: write video/<save_file>.gif and .mp4 (the `video` folder must exist; `save_file`
          should not contain dots, they are treated as a file extension)
    show: call plt.show()
    rotate, n_rotations: slowly rotate the camera by n_rotations full turns during the animation
    show_progress: True - trails grow frame by frame; False - the whole path is drawn from the start
    elev: camera elevation angle in degrees

    Returns: the FuncAnimation object.
    """
    glider_arr = np.asarray(glider_history, dtype=np.float32)
    target_arr = np.asarray(target_history, dtype=np.float32)

    X_g, Y_g, Z_g = glider_arr[:, 0], glider_arr[:, 1], glider_arr[:, 2]
    X_t, Y_t, Z_t = target_arr[:, 0], target_arr[:, 1], target_arr[:, 2]

    all_x = np.concatenate([X_g, X_t])
    all_y = np.concatenate([Y_g, Y_t])

    # Plot limits: at least [-5.5, 5.5], expanded to fit both trajectories.
    lim_xy = max(
        5.0,
        float(np.abs(all_x).max()),
        float(np.abs(all_y).max())
    ) + 0.5

    # Coarser grid for large areas to keep rendering fast.
    grid_size = 60 if lim_xy < 10 else 40

    x = np.linspace(-lim_xy, lim_xy, grid_size)
    y = np.linspace(-lim_xy, lim_xy, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = Z_func(X, Y)

    # Vertical limits cover the surface and both trajectories, with a small margin.
    z_min = float(min(Z_g.min(), Z_t.min(), Z.min())) - 0.2
    z_max = float(max(Z_g.max(), Z_t.max(), Z.max())) + 0.2

    fig = plt.figure(figsize=(9, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Respect the explicit zorder values below (matplotlib would otherwise sort by depth),
    # so that trails and markers are drawn on top of the surface.
    ax.computed_zorder = False

    # The surface (semi-transparent so the trajectories stay visible).
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6,
                           edgecolor='none', zorder=1)

    # Trails.
    line_g, = ax.plot([], [], [], color="tab:blue", linewidth=2,
                      label="Glider", zorder=10)
    line_t, = ax.plot([], [], [], color="tab:red", linewidth=2,
                      linestyle="--", label="Target", zorder=10)

    # Current positions (no legend entries).
    dot_g, = ax.plot([], [], [], "o", color="tab:blue", markersize=10,
                     markeredgecolor="white", markeredgewidth=1.5,
                     label="_nolegend_", zorder=11)
    dot_t, = ax.plot([], [], [], "o", color="tab:red", markersize=10,
                     markeredgecolor="white", markeredgewidth=1.5,
                     label="_nolegend_", zorder=11)

    ax.set_xlim(-lim_xy, lim_xy)
    ax.set_ylim(-lim_xy, lim_xy)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Glider and target on manifold (3D)")
    ax.legend(loc="upper left")

    ax.view_init(elev=elev, azim=45)

    n_frames = max(1, len(X_g))

    # Lift the dots slightly above the surface so they are not hidden by it.
    z_eps = 0.01 * (z_max - z_min)

    def init():
        """Start of the animation: empty trails and no dots."""
        line_g.set_data_3d([], [], [])
        line_t.set_data_3d([], [], [])
        dot_g.set_data_3d([], [], [])
        dot_t.set_data_3d([], [], [])
        return line_g, line_t, dot_g, dot_t

    def update(frame):
        """Draw the state at step `frame` and rotate the camera if requested."""
        if show_progress:
            # Trails grow with the animation.
            i = min(frame, len(X_g) - 1)
            line_g.set_data_3d(X_g[:i + 1], Y_g[:i + 1], Z_g[:i + 1])
            line_t.set_data_3d(X_t[:i + 1], Y_t[:i + 1], Z_t[:i + 1])
            dot_g.set_data_3d([X_g[i]], [Y_g[i]], [Z_g[i] + z_eps])
            dot_t.set_data_3d([X_t[i]], [Y_t[i]], [Z_t[i] + z_eps])
        else:
            # Whole paths from the first frame, dots at the final positions.
            line_g.set_data_3d(X_g, Y_g, Z_g)
            line_t.set_data_3d(X_t, Y_t, Z_t)
            dot_g.set_data_3d([X_g[-1]], [Y_g[-1]], [Z_g[-1] + z_eps])
            dot_t.set_data_3d([X_t[-1]], [Y_t[-1]], [Z_t[-1] + z_eps])

        if rotate:
            # Azimuth goes from 45 degrees through n_rotations full turns over the animation.
            azim = 45.0 + (frame / n_frames) * 360.0 * n_rotations
            ax.view_init(elev=elev, azim=azim)

        return line_g, line_t, dot_g, dot_t

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=30,
        blit=False,  # blitting does not work with 3D axes / camera rotation
        repeat=True,
    )

    # with the "Agg" backend selected above, plt.show() does not open a window.
    if show:
        plt.show()

    if save:
        PATH = BASE_DIR.parent / "video" / save_file
        anim.save(PATH.with_suffix(".gif"), writer="pillow", fps=30, dpi=100)
        anim.save(PATH.with_suffix(".mp4"), writer="ffmpeg", fps=30, dpi=100)

    return anim