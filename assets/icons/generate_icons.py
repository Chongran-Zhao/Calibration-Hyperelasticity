import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 3
plt.rcParams["lines.linewidth"] = 5
plt.rcParams["lines.markersize"] = 12


def generate_scientific_icon(output_path="icon_scientific.png"):
    """
    Scientific icon: a smooth hyperelastic fit curve through scattered data.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    lam = np.linspace(1, 2.5, 100)
    stress = (lam - lam**(-2)) * 5

    lam_exp = np.linspace(1.1, 2.4, 6)
    stress_exp = (lam_exp - lam_exp**(-2)) * 5 + np.random.normal(0, 0.5, size=len(lam_exp))

    ax.plot(lam, stress, color="#007AFF", alpha=0.9, zorder=2)
    ax.plot(
        lam_exp,
        stress_exp,
        "o",
        markerfacecolor="white",
        markeredgecolor="#FF9500",
        markeredgewidth=3,
        zorder=3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.set_xticks([])
    ax.set_yticks([])

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close()


def generate_abstract_icon(output_path="icon_abstract.png"):
    """
    Abstract icon: a rounded container with a smooth calibration curve.
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])

    circle = Circle((0.5, 0.5), 0.48, color="#F0F0F0", alpha=1.0, zorder=0)
    ax.add_patch(circle)

    t = np.linspace(0.2, 0.8, 100)
    y = 0.5 + 0.3 * np.tanh(10 * (t - 0.5))

    ax.plot(t, y, color="#D0021B", lw=8, zorder=2)
    ax.scatter(
        [0.35, 0.65],
        [0.5 + 0.3 * np.tanh(10 * (0.35 - 0.5)), 0.5 + 0.3 * np.tanh(10 * (0.65 - 0.5))],
        s=300,
        color="#333333",
        zorder=3,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_alpha(0.0)

    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()


if __name__ == "__main__":
    generate_scientific_icon()
    generate_abstract_icon()
