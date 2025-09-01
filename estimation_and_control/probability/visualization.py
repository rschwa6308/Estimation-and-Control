from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch
import numpy as np

from probability.distributions import HistogramDistribution

# from helpers import query_pdf


def plot_pdf_values(ax: plt.Axes, sample_points: np.ndarray, pdf_values: np.ndarray, transpose_axis=False, edgecolor="black", **kwargs):
    "Plot a PDF evaluated at sample points"
    assert(sample_points.shape == pdf_values.shape)

    if not transpose_axis:
        ax.fill_between(sample_points, pdf_values, edgecolor=edgecolor, **kwargs)
        ax.set_ylim(0, None)

    else:
        ax.fill_betweenx(sample_points, pdf_values,edgecolor=edgecolor, **kwargs)
        ax.set_xlim(0, None)


def plot_pdf_transform(prior: HistogramDistribution, transform_values, posterior: HistogramDistribution, slice_highlights=[], highlight_color="tab:orange", prior_label=None, posterior_label=None, transform_label=None, title=None, figsize=(6, 6)):
    if prior.dim > 1 or posterior.dim > 1:
        raise NotImplementedError("Plotting PDF transform for dimensions > 1 not supported.")

    fig, axes = plt.subplots(
        2, 2, figsize=figsize,
        sharex="col", sharey="row",
        constrained_layout=True
    )
    axes[1, 0].axis("off")

    input_domain = prior.domain[:,0]
    output_domain = posterior.domain[:,0]

    input_samples = prior.bin_midpoints[:,0]
    output_samples = posterior.bin_midpoints[:,0]

    input_values = prior.pdf_values[:]
    output_values = posterior.pdf_values[:]

    # plot PDFs
    plot_pdf_values(axes[1, 1], input_samples, input_values, label=prior_label, zorder=2)
    plot_pdf_values(axes[0, 0], output_samples, output_values, transpose_axis=True, label=posterior_label, zorder=2)

    # plot transform
    axes[0, 1].plot(input_domain, transform_values, label=transform_label)

    # set PDF axis limits explicitly so they match
    pmax = max(
        np.quantile(input_values, 0.95),
        np.quantile(output_values, 0.95)
    ) * 1.2
    axes[0, 0].set_xlim(0, pmax)
    axes[1, 1].set_ylim(0, pmax)

    # EXPERIMENTAL: make output PDF axis log scale
    # axes[0, 0].set_xscale("functionlog", functions=[lambda x: 1 + 100*x, lambda x: (x - 1)/100])
    # axes[1, 1].set_yscale("functionlog", functions=[lambda x: 1 + 100*x, lambda x: (x - 1)/100])

    # set PDF axis ticks explicitly so they match
    axes[1, 1].set_xticks(np.linspace(input_domain[0], input_domain[-1], 5))
    axes[0, 0].set_yticks(np.linspace(output_domain[0], output_domain[-1], 5))

    # rotate posterior PDF tick labels
    axes[0, 0].tick_params(axis="y", rotation=-90)

    # add subplot captions
    axes[1, 1].set_xlabel(r"$\mathrm{bel}(x_t)$")
    axes[0, 0].set_ylabel(r"$\mathrm{bel}(x_{t+1})$", rotation=-90, va="top")

    # optional: set figure title
    if title is not None:
        fig.suptitle(title)

    # optional: make legends
    if prior_label is not None:     axes[1, 1].legend(loc="upper right")
    if posterior_label is not None: axes[0, 0].legend(loc="upper right")
    if transform_label is not None: axes[0, 1].legend(loc="upper right")
    

    # for given slices of the domain, highlight the preimage and image under the transform
    for x_low, x_high in slice_highlights:
        domain_step = input_domain[1] - input_domain[0]
        x_low_idx = int((x_low - input_domain[0]) / domain_step)
        x_high_idx = int((x_high - input_domain[0]) / domain_step)

        y_low = transform_values[x_low_idx, 0]
        y_high = transform_values[x_high_idx, 0]

        axes[1,1].add_artist(ConnectionPatch(
            (x_low, 0), (x_low, y_low),
            coordsA="data", coordsB="data",
            axesA=axes[1, 1], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8,
            zorder=1
        ))

        axes[1,1].add_artist(ConnectionPatch(
            (x_high, 0), (x_high, y_high),
            coordsA="data", coordsB="data",
            axesA=axes[1, 1], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8,
            zorder=1
        ))

        axes[0,0].add_artist(ConnectionPatch(
            (0, y_low), (x_low, y_low),
            coordsA="data", coordsB="data",
            axesA=axes[0, 0], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8,
            zorder=1
        ))

        axes[0,0].add_artist(ConnectionPatch(
            (0, y_high), (x_high, y_high),
            coordsA="data", coordsB="data",
            axesA=axes[0, 0], axesB=axes[0, 1],
            color=highlight_color, linestyle="--", linewidth=0.8,
            zorder=1
        ))


        mask = (x_low <= input_samples) & (input_samples <= x_high)
        plot_pdf_values(
            axes[1, 1], input_samples[mask], input_values[mask],
            linestyle="--", alpha=0.9, linewidth=0.8, facecolor=highlight_color,
            zorder=2
        )

        # enforce y_low <= y_high
        y_low, y_high = min(y_low, y_high), max(y_low, y_high)

        mask = (y_low <= output_samples) & (output_samples <= y_high)
        plot_pdf_values(
            axes[0, 0], output_samples[mask], output_values[mask],
            transpose_axis=True,
            linestyle="--", alpha=0.9, linewidth=0.8, facecolor=highlight_color,
            zorder=2
        )

    # change subplot draw order so that the connection patches draw on top of the transform plot
    axes[0, 0].set_zorder(1)
    axes[0, 1].set_zorder(0)
    axes[1, 1].set_zorder(1)

    return axes

import matplotlib.patches
import matplotlib.transforms


def plot_covariance_ellipse(ax, mean, cov, n_std=1.0, edgecolor="black", **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this two-dimensional distribution
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        edgecolor=edgecolor, **kwargs
    )

    mean_x, mean_y = mean.flatten()

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = matplotlib.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
