"""Plot physical density fields without changing the saved design."""

from pathlib import Path
import numpy as np


def plot_design(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    directory, output = Path(directory), Path(output)
    rho = np.load(directory / "rho.npy", allow_pickle=False)
    if not (directory / "protocol.json").exists():
        with np.load(directory / "geometry.npz", allow_pickle=False) as data:
            coords, cells = data["coords"], data["cells"]
    else:
        coords = np.load(directory / "coords.npy", allow_pickle=False)
        cells = np.load(directory / "econn.npy", allow_pickle=False)
    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    if coords.shape[1] == 2:
        ax = fig.add_subplot()
        ax.add_collection(
            PolyCollection(
                coords[cells],
                array=rho,
                cmap="gray_r",
                clim=(0, 1),
                edgecolors="none",
                rasterized=True,
            )
        )
        ax.autoscale_view()
        ax.set_aspect("equal")
    else:
        from skimage.measure import marching_cubes
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        centers = coords[cells].mean(axis=1)
        axes = [np.unique(centers[:, k]) for k in range(3)]
        volume = np.zeros(tuple(len(a) for a in axes))
        indexes = tuple(np.searchsorted(a, centers[:, k]) for k, a in enumerate(axes))
        volume[indexes] = rho
        spacing = [a[1] - a[0] for a in axes]
        vertices, faces, _, _ = marching_cubes(np.pad(volume, 1), level=0.5, spacing=spacing)
        ax = fig.add_subplot(projection="3d")
        ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor="0.55", linewidth=0))
        ax.auto_scale_xyz(*vertices.T)
        ax.set_box_aspect(np.ptp(vertices, axis=0))
        ax.view_init(elev=22, azim=-65)
    ax.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
