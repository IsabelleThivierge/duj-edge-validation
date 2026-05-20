import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# CONFIG
# -------------------------

N_AGENTS = 10000
WIDTH = 100
HEIGHT = 100
FRAMES = 300

DRIFT_THRESHOLD = 0.12
CORRECTION_THRESHOLD = 0.22

np.random.seed(42)

# -------------------------
# AGENT STATE
# -------------------------

x = np.random.uniform(0, WIDTH, N_AGENTS)
y = np.random.uniform(0, HEIGHT, N_AGENTS)

vx = np.random.normal(0, 0.15, N_AGENTS)
vy = np.random.normal(0, 0.15, N_AGENTS)

residual = np.random.uniform(0, 0.05, N_AGENTS)

correction_flash = np.zeros(N_AGENTS)

# -------------------------
# FIGURE
# -------------------------

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)

ax.set_xticks([])
ax.set_yticks([])

scatter = ax.scatter(
    x,
    y,
    s=4,
    c="lime",
    alpha=0.8
)

title = ax.set_title(
    "DUJ Living Swarm",
    color="white"
)

# -------------------------
# UPDATE
# -------------------------

def update(frame):

    global x, y, residual

    residual += np.random.normal(
        0,
        0.005,
        N_AGENTS
    )

    x[:] = (x + vx + residual * 0.3) % WIDTH
    y[:] = (y + vy + residual * 0.3) % HEIGHT

    violating = residual > CORRECTION_THRESHOLD

    correction_flash[:] = 0

    corrected = np.where(violating)[0]

    if len(corrected) > 0:
        residual[corrected] *= 0.55
        correction_flash[corrected] = 1

    colors = np.full(
        N_AGENTS,
        "lime",
        dtype=object
    )

    drifting = (
        (residual > DRIFT_THRESHOLD)
        & (residual <= CORRECTION_THRESHOLD)
    )

    colors[drifting] = "yellow"
    colors[violating] = "red"
    colors[correction_flash == 1] = "white"

    scatter.set_offsets(
        np.column_stack((x, y))
    )

    scatter.set_color(colors)

    title.set_text(
        f"DUJ Living Swarm | step={frame}"
    )

    return scatter,

# -------------------------
# SAVE ANIMATION
# -------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=FRAMES,
    interval=50,
    blit=False
)

print("Saving animation...")

ani.save(
    "duj_swarm.gif",
    fps=20
)

print("Saved: duj_swarm.gif")
