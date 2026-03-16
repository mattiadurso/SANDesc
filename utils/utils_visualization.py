from __future__ import annotations
from typing import Tuple

import warnings
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import matplotlib.patheffects as PathEffects
import numpy as np
import torch
from torch import Tensor
import math
from utils.utils_image import pad_and_cut_image, cat_images


import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from torch import Tensor

from utils.utils_homography import warp_points
from utils.utils_keypoints import (
    find_distance_matrices_between_points_and_their_projections,
)
from utils.utils_matches import MatchesWithExtra

# filter out the useless matplotlib warnings
warnings.filterwarnings("ignore", "This figure includes Axes")


def subplots(
    ny: int = 1,
    nx: int = 1,
    n: int | None = None,
    figsize: Tuple[int, int] | None = None,
    dpi: float | None = None,
    name: str = "plot",
    title: str | None = None,
    axes_visible: bool = False,
    hspace: float = 0.001,
    wspace: float = 0.001,
    subplots_adjust: bool = True,
) -> Tuple[plt.Figure, np.ndarray | matplotlib.axes.Axes]:
    if n is not None:
        nx = math.ceil(math.sqrt(n))
        ny = math.ceil(n / nx)

    # initialize matplotlib stuff
    fig = plt.figure(figsize=figsize, dpi=dpi, num=name)
    # use gridspec to remove the empty borders between subplots
    gs = fig.add_gridspec(ny, nx, hspace=hspace, wspace=wspace)
    axes = gs.subplots()
    if subplots_adjust:
        # reduce the outer spacing between subplots
        fig.subplots_adjust(
            left=wspace / nx,
            right=1 - wspace / nx,
            top=1 - hspace / ny,
            bottom=hspace / ny,
        )
    # remove ticks and labels
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    if isinstance(axes, np.ndarray):
        if not axes_visible:
            for ax in axes.flatten():
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.axis("off")
                ax.set_aspect("equal")
    if title is not None:
        fig.suptitle(title)
    return fig, axes


def imshow(
    img: Tensor | list[Tensor] | np.ndarray | list[np.ndarray],
    ax: plt.Axes | None = None,
    title: str = "",
    cmap: str = "inferno",
    figsize: Tuple[int, int] = (15, 15),
    block: bool = False,
    pad: int = 5,
    tight_layout: bool = True,
    plot_values: bool = False,
    nx: int | None = None,
    show: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    axis_off: bool = False,
    alpha: float = 1.0,
    name: str = "plot",
) -> any:
    """wrapper for plt.imshow(). It can get a single image or a list, and it finds the best way to display them
        as a single concatenated image

    Returns:
        object:
    """
    if ax is None:
        fig, handler = subplots(1, 1, figsize=figsize, name=name)
        handler = handler[0]
    else:
        handler = ax

    if axis_off:
        handler.get_xaxis().set_visible(False)
        handler.get_yaxis().set_visible(False)
        handler.axis("off")
        handler.set_aspect("equal")

    def image_to_tensor(x: np.ndarray | Tensor) -> Tensor:
        """convert image to tensor if it was a numpy array"""
        if isinstance(x, np.ndarray):
            # convert the list of np.ndarray to list of Tensor
            assert x.ndim == 2 or x.ndim == 3
            if x.ndim == 3:
                return torch.tensor(x).permute(2, 0, 1)
            else:
                return torch.tensor(x)
        else:
            return x.detach()

    if isinstance(img, list):
        # input is a list of images
        imgs = [image_to_tensor(x.detach()) for x in img]

        n_imgs = len(imgs)
        n_x = math.ceil(math.sqrt(n_imgs)) if nx is None else nx
        n_y = math.ceil(n_imgs / n_x)

        H_max_per_row = []
        W_max_per_column = []
        for i in range(n_y):
            H_max = 0
            for j in range(n_x):
                if i * n_x + j < n_imgs:
                    H_max = max(H_max, imgs[i * n_x + j].shape[-2])
            H_max_per_row.append(H_max)

        for j in range(n_x):
            W_max = 0
            for i in range(n_y):
                if i * n_x + j < n_imgs:
                    W_max = max(W_max, imgs[i * n_x + j].shape[-1])
            W_max_per_column.append(W_max)

        imgs_row = []
        for i in range(n_y):
            img_row = pad_and_cut_image(
                imgs[i * n_x], (H_max_per_row[i] + pad, W_max_per_column[0] + pad)
            )
            for j in range(1, n_x):
                if i * n_x + j < n_imgs:
                    img_pad = pad_and_cut_image(
                        imgs[i * n_x + j],
                        (H_max_per_row[i] + pad, W_max_per_column[j] + pad),
                    )
                    img_row = cat_images([img_row, img_pad], mode="horizontal")
            imgs_row.append(img_row)

        img_out = imgs_row[0]
        for i in range(1, len(imgs_row)):
            img_out = cat_images([img_out, imgs_row[i]], mode="vertical")
    else:
        img_out = image_to_tensor(img)

    if img_out.ndim == 2:
        ref = handler.imshow(
            img_out.cpu(),
            cmap=cmap,
            extent=[0, img_out.shape[1], img_out.shape[0], 0],
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )
    else:
        ref = handler.imshow(
            img_out.permute(1, 2, 0).cpu(),
            cmap=cmap,
            extent=[0, img_out.shape[2], img_out.shape[1], 0],
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )

    if plot_values:
        texts = []
        for i in range(img_out.shape[0]):
            for j in range(img_out.shape[1]):
                if ~img_out[i, j].isnan():
                    txt = handler.text(
                        j + 0.5,
                        i + 0.5,
                        f"{img_out[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="green",
                    )
                    texts.append(txt)

    if title:
        handler.set_title(title)
    if tight_layout:
        plt.tight_layout()
    if show:
        plt.show(block=block)
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()

    return ref


def scatter(
    xy: Tensor,
    radius: Tensor | float = 5.0,
    c: str | Tensor = "b",
    label: str = None,
    ax: plt.Axes = None,
    real_dimensions: bool = False,
    linewidth: Tensor | float = 2.0,
    marker: str = None,
    alpha: float = 1.0,
    mask_outside: bool = False,
    texts: list[str] = None,
    texts_fontsize: int = 20,
    cmap: str = None,
    normalize_color: bool = True,
    border_color: str = None,
    border_width: float = 3.0,
    full: bool = False,
) -> None:
    """wrapper for the scatter function, the s parameter is the radius of the scatter
    Args:
        xy: the coordinates of the points to be plotted (x, y) with convention top-left pixel center at (0.5, 0.5)
            n,2
        radius: the radius of the scatter. Can be either one single value or a value for each point
        c: the color of the scatter
        label: the label of the scatter plot
        ax: the axis where to plot the scatter
        real_dimensions: if True, the radius correspond to real pixel dimensions
        linewidth: the linewidth of the scatter
        marker: the marker of the scatter
        alpha: the alpha of the scatter
        mask_outside: if True, only the points that fall inside the axis are plotted
        texts: if provided, additionally plot the text for each point
        texts_fontsize: the fontsize of the text
        cmap: the colormap to use for the points
        normalize_color: if True, normalize the color between 0 and 1
        border_color: if not none, apply a border to the points
        border_width: the width of the border
        full: if True, plot the markers as full
    """
    if xy.ndim == 1:
        xy = xy[None]
    if isinstance(radius, Tensor):
        assert radius.ndim == 1
        assert xy.shape[0] == radius.shape[0]
        radius = radius.cpu().numpy()

    if isinstance(linewidth, Tensor):
        assert linewidth.ndim == 1
        assert xy.shape[0] == linewidth.shape[0]
        linewidth = linewidth.cpu().numpy()

    if isinstance(c, Tensor):
        assert xy.shape[0] == c.shape[0]
        c = c.cpu().numpy()

    if texts is not None:
        assert xy.shape[0] == len(texts)

    if cmap is not None:
        assert isinstance(
            c, np.ndarray
        ), "c must be a numpy array when using a colormap"
        if normalize_color:
            c = (c - c.min(initial=None)) / (c.max(initial=None) - c.min(initial=None))
        c = plt.get_cmap(cmap)(c)

    if mask_outside:
        x_max = ax.get_xlim()[1]
        y_max = ax.get_ylim()[0]
        # mask the point that project outside
        mask_inside = (0 < xy).all(-1) * (xy[:, 0] < x_max) * (xy[:, 1] < y_max)
        xy = xy[mask_inside]
        if isinstance(radius, np.ndarray):
            radius = radius[mask_inside]
        if isinstance(linewidth, np.ndarray):
            linewidth = linewidth[mask_inside]

    handler = plt.gca() if ax is None else ax

    if border_color is not None:
        path_effects = [
            PathEffects.withStroke(linewidth=border_width, foreground=border_color)
        ]
    else:
        path_effects = []

    if real_dimensions:
        # plot circle that does not chance dimension when zooming in
        assert xy.ndim == 2
        if isinstance(radius, float):
            radius = torch.ones(xy.shape[0]) * radius
        for x, y, r, color in zip(xy[:, 0].cpu(), xy[:, 1].cpu(), radius, c):
            handler.add_artist(
                Circle(xy=(x, y), radius=r, facecolor="none", edgecolor=color)
            )
        handler.scatter(
            xy[:, 0].cpu(),
            xy[:, 1].cpu(),
            marker="+",
            s=20,
            c=c,
            linewidths=1,
            label=label,
            alpha=alpha,
        )
    else:
        if marker in {"+", "x"} or full:
            facecolors = c
            edgecolors = None
        else:
            facecolors = "none"
            edgecolors = c

        handler.scatter(
            xy[..., 0].cpu(),
            xy[..., 1].cpu(),
            edgecolors=edgecolors,
            facecolors=facecolors,
            s=math.pi * radius**2,
            label=label,
            linewidths=linewidth,
            marker=marker,
            alpha=alpha,
            path_effects=path_effects,
        )

    if texts is not None:
        for color, xy_, text_ in zip(c, xy, texts):
            handler.annotate(
                f"{text_}",
                xy=xy_,
                color=color,
                ha="center",
                va="center",
                fontsize=texts_fontsize,
            )


def fill(
    xy: Tensor,
    c: str | Tensor = "b",
    label: str | None = None,
    ax: plt.Axes | None = None,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    border_color: str | None = None,
    border_width: float = 3.0,
) -> None:
    """wrapper for the scatter function, the s parameter is the radius of the scatter
    Args:
        xy: the coordinates of the points to be plotted (x, y) with convention top-left pixel center at (0.5, 0.5)
            n,2
        c: the color of the scatter
        label: the label of the scatter plot
        ax: the axis where to plot the scatter
        linewidth: the linewidth of the scatter
        alpha: the alpha of the scatter
        border_color: if not none, apply a border to the points
        border_width: the width of the border
    """
    if xy.ndim == 1:
        xy = xy[None]

    if isinstance(c, Tensor):
        c = c.cpu().numpy()
    elif isinstance(c, float):
        c = np.array([c]).repeat(xy.shape[0])
    elif isinstance(c, str):
        pass

    handler = plt.gca() if ax is None else ax

    if border_color is not None:
        path_effects = [
            PathEffects.withStroke(linewidth=border_width, foreground=border_color)
        ]
    else:
        path_effects = []

    handler.fill(
        xy[..., 0].cpu(),
        xy[..., 1].cpu(),
        edgecolor=c,
        facecolor="none",
        linewidth=linewidth,
        label=label,
        alpha=alpha,
        path_effects=path_effects,
    )


def text(
    string: str,
    xy: Tensor | np.ndarray | Tuple[int, int],
    c: str | Tensor = "r",
    fontsize: int = 14,
    ha: str = "left",
    va: str = "top",
    border_color: str = None,
    border_width: int = 2,
    alpha: float = 1.0,
    ax: plt.Axes = None,
) -> any:
    """wrapper for the fill function
    Args:
        string: the text to be plotted
        xy: coordinates of the text
        c: the color of the fill
        fontsize: the fontsize of the text
        ha: the horizontal alignment of the text
        va: the vertical alignment of the text
        border_color: the color of the border of the text
        border_width: the width of the border of the text
        alpha: the alpha of the scatter
        ax: the axis where to plot the fill
    """
    if isinstance(xy, Tensor):
        xy = xy.cpu().numpy()

    handler = plt.gca() if ax is None else ax

    if border_color is not None:
        path_effects = [
            PathEffects.withStroke(linewidth=border_width, foreground=border_color)
        ]
    else:
        path_effects = None

    ref = handler.text(
        xy[0],
        xy[1],
        string,
        color=c,
        va=va,
        ha=ha,
        fontsize=fontsize,
        alpha=alpha,
        path_effects=path_effects,
        clip_on=True,
    )
    return ref


def plot_pairs(
    xy0: Tensor,
    xy1: Tensor,
    c=None,
    linewidth=None,
    ax=None,
):
    """plot a line for each xy pair provided as input
    inputs:
        xy0: first set of points
            n,2
        xy1: second set of points
            n,2
    """
    assert xy0.shape == xy1.shape, f"xy0.shape = {xy0.shape}, xy1.shape = {xy1.shape}"
    handler = plt if ax is None else ax

    lines = [np.array(points) for points in zip(xy0.cpu().numpy(), xy1.cpu().numpy())]
    lc = LineCollection(lines, colors=c, linewidths=linewidth)
    handler.add_collection(lc)


def plot_gaussian_ellipses(
    mean: Tensor,
    covariance: Tensor,
    c: str | Tensor = "b",
    ax: plt.Axes = None,
    linewidth: float = 1.0,
    alpha: float = 1.0,
    border_color: str = None,
    border_width: float = 3.0,
    linestyle: str = None,
) -> None:
    """
    Args:
        mean: the mean of the gaussian
            2
        covariance: the covariance of the gaussian
            single float | Tensor 1 | Tensor 2 | Tensor 2,2
        c: the color of the scatter
        ax: the axis where to plot the scatter
        linewidth: the linewidth of the scatter
        alpha: the alpha of the scatter
        border_color: if not none, apply a border to the points
        border_width: the width of the border
        linestyle: linestyle of the ellipse
    """
    handler = plt.gca() if ax is None else ax

    if border_color is not None:
        path_effects = [
            PathEffects.withStroke(linewidth=border_width, foreground=border_color)
        ]
    else:
        path_effects = []

    if isinstance(covariance, float) or covariance.numel() == 1:
        if isinstance(covariance, Tensor):
            covariance = covariance.item()
        diam = 2 * np.sqrt(covariance)  # two times sigma

        for factor in [1, 2, 3]:
            ellipse_diam = factor * diam
            ellipse = matplotlib.patches.Ellipse(
                mean.cpu().numpy(),
                ellipse_diam,
                ellipse_diam,
                angle=0,
                edgecolor=c,
                facecolor="none",
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            if path_effects:
                ellipse.set_path_effects(path_effects)
            handler.add_artist(ellipse)
    else:
        raise NotImplementedError


def plot_rectangle(
    ax,
    center: Tensor,
    dimensions: Tensor,
    max_dimensions: Tuple[int, int] = None,
    c: str = "r",
):
    """plot a rectangle centered in center, if max_dimensions is provided, the plot is clamped to stay i the image
    Args:
        ax: the matplotlib axis to use for plotting
        center: the center coordinate given as (x, y)
            2
        dimensions: the rectangle dimensions given as (W, H)
            2
        max_dimensions: (W, H) if provided the plot remain inside the image
        c: the color
    Returns:
        ax_plot
    """
    corners_for_plot = torch.tensor(
        [
            [center[0] - dimensions[0] / 2, center[1] - dimensions[1] / 2],
            [center[0] - dimensions[0] / 2, center[1] + dimensions[1] / 2],
            [center[0] + dimensions[0] / 2, center[1] + dimensions[1] / 2],
            [center[0] + dimensions[0] / 2, center[1] - dimensions[1] / 2],
            [center[0] - dimensions[0] / 2, center[1] - dimensions[1] / 2],
        ],
        device="cpu",
    )
    corners_for_plot[:, 0] = corners_for_plot[:, 0].clamp(0, max_dimensions[0])
    corners_for_plot[:, 1] = corners_for_plot[:, 1].clamp(0, max_dimensions[1])
    return ax.plot(corners_for_plot[:, 0], corners_for_plot[:, 1], c=c)


def plot_image_pair_with_keypoints_repeatability(
    img0: Tensor | np.ndarray,
    img1: Tensor | np.ndarray,
    xy0: Tensor,
    xy1: Tensor,
    hom: Tensor,
    radius: float = 5,
    name: str = None,
    title: str = None,
    figsize: tuple[int, int] = None,
    dpi: float = None,
    marker: str | None = None,
    axes: np.ndarray = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Args:
        img0: image 0
        img1: image 1
        xy0: keypoints in image 0 (x, y) with convention top-left pixel center at coordinate (0.5, 0.5)
            n0,2
        xy1: keypoints in image 1 (x, y) with convention top-left pixel center at coordinate (0.5, 0.5)
            n1,2
        hom: the homography that links the two images
        radius: the radius of the scatter
        name: the name of the figure
        title: the plot title
        figsize: the figure size
        dpi: the figure dpi
        axes: if provided, do not open a new figure but instead plot on the provided axes
    """
    assert img0.ndim in {2, 3} and img1.ndim in {2, 3}, "image.ndim must be 2 or 3"
    assert xy0.ndim == 2 and xy1.ndim == 2, "xy must be a 2D tensor"
    assert xy0.shape[1] == 2 and xy1.shape[1] == 2, "xy must have 2 columns"
    assert hom.shape == (3, 3), "hom must be a 3,3 tensor"

    if axes is None:
        fig, axes = subplots(1, 2, figsize=figsize, dpi=dpi, name=name)
        fig.suptitle(title) if title is not None else None
    else:
        fig = axes[0].figure
        assert axes.shape == (2,)

    imshow(img0, ax=axes[0], show=False, cmap="gray")
    imshow(img1, ax=axes[1], show=False, cmap="gray")

    xy0_proj = warp_points(xy0[None], hom[None], img1.shape[-2:])[0].to(
        torch.float
    )  # n0,2
    xy1_proj = warp_points(xy1[None], torch.inverse(hom)[None], img0.shape[-2:])[0].to(
        torch.float
    )  # n1,2

    if xy0.shape[0] > 0 and xy1.shape[0] > 0:
        dist0, dist1 = find_distance_matrices_between_points_and_their_projections(
            xy0, xy1, xy0_proj, xy1_proj
        )
        xy0_dist = dist1.min(-1)[0]  # n0
        xy1_dist = dist0.min(-2)[0]  # n1

        xy0_color = torch.tensor([1.0, 0.0, 0.0])[None].repeat(
            xy0.shape[0], 1
        )  # set everything as red
        xy0_color[xy0_dist.isinf()] = torch.tensor(
            [0.0, 0.0, 1.0]
        )  # blue for the invalid depth or project out
        xy0_color[xy0_dist < 3.0] = torch.tensor(
            [1.0, 0.5, 0.0]
        )  # orange for 2 <= dist < 3
        xy0_color[xy0_dist < 2.0] = torch.tensor(
            [1.0, 1.0, 0.0]
        )  # yellow for 2 <= dist < 3
        xy0_color[xy0_dist < 1.0] = torch.tensor(
            [0.0, 1.0, 0.0]
        )  # green for 2 <= dist < 3
        xy1_color = torch.tensor([1.0, 0.0, 0.0])[None].repeat(
            xy1.shape[0], 1
        )  # set everything as red
        xy1_color[xy1_dist.isinf()] = torch.tensor(
            [0.0, 0.0, 1.0]
        )  # blue for the invalid depth or project out
        xy1_color[xy1_dist < 3.0] = torch.tensor(
            [1.0, 0.5, 0.0]
        )  # orange for 2 <= dist < 3
        xy1_color[xy1_dist < 2.0] = torch.tensor(
            [1.0, 1.0, 0.0]
        )  # yellow for 2 <= dist < 3
        xy1_color[xy1_dist < 1.0] = torch.tensor(
            [0.0, 1.0, 0.0]
        )  # green for 2 <= dist < 3
    else:
        xy0_color = "r"
        xy1_color = "r"

    scatter(xy0, ax=axes[0], radius=radius, c=xy0_color, linewidth=1, marker=marker)
    scatter(
        xy0_proj, ax=axes[1], radius=radius / 2, c=xy0_color, linewidth=1, marker=marker
    )
    scatter(xy1, ax=axes[1], radius=radius, c=xy1_color, linewidth=1, marker=marker)
    scatter(
        xy1_proj, ax=axes[0], radius=radius / 2, c=xy1_color, linewidth=1, marker=marker
    )
    return fig, axes


def plot_image_pair_with_keypoints_and_matches(
    img0: Tensor | np.ndarray,
    img1: Tensor | np.ndarray,
    xy0: Tensor,
    xy1: Tensor,
    matches: Tensor,
    hom: Tensor,
    highlight_mask: Tensor | None = None,
    show_all_keypoints: bool = True,
    radius: float = 5,
    matches_alpha: float = 0.5,
    name: str | None = None,
    title: str = None,
    figsize: tuple[int, int] = None,
    dpi: float = None,
    axes: np.ndarray = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Args:
        img0: image 0
        img1: image 1
        xy0: keypoints in image 0 (x, y) with convention top-left pixel center at coordinate (0.5, 0.5)
            n0,2
        xy1: keypoints in image 1 (x, y) with convention top-left pixel center at coordinate (0.5, 0.5)
            n1,2
        matches: matches formatted as (index of matched xy0, index of matched xy1)
            n_matches, 2
        hom: the homography that links the two images
        highlight_mask: if provided, the matches are plotted thicker
        radius: the radius of the scatter
        matches_alpha: the alpha of the matches
        show_all_keypoints: if True, plot all the keypoints
        name: the name of the figure
        title: the plot title
        figsize: the figure size
        dpi: the figure dpi
        axes: if provided, do not open a new figure but instead plot on the provided axes
    """
    assert img0.ndim in {2, 3} and img1.ndim in {2, 3}, "image.ndim must be 2 or 3"
    assert xy0.ndim == 2 and xy1.ndim == 2, "xy must be a 2D tensor"
    assert xy0.shape[1] == 2 and xy1.shape[1] == 2, "xy must have 2 columns"
    assert hom.shape == (3, 3), "hom must be a 3,3 tensor"

    if axes is None:
        fig, axes = subplots(1, 2, figsize=figsize, dpi=dpi, name=name)
        fig.suptitle(title) if title is not None else None
    else:
        fig = axes[0].figure
        assert axes.shape == (2,)

    imshow(img0, ax=axes[0], show=False)
    imshow(img1, ax=axes[1], show=False)

    xy0_proj = warp_points(xy0[None], hom[None], img1.shape[-2:])[0].to(
        torch.float
    )  # n0,2
    xy1_proj = warp_points(xy1[None], torch.inverse(hom)[None], img0.shape[-2:])[0].to(
        torch.float
    )  # n1,2

    xy0_matched = xy0[matches[:, 0]]
    xy1_matched = xy1[matches[:, 1]]

    xy0_proj_matched = xy0_proj[matches[:, 0]]
    xy1_proj_matched = xy1_proj[matches[:, 1]]

    dist_in_img0 = torch.norm(xy0_matched - xy1_proj_matched, dim=-1)
    dist_in_img1 = torch.norm(xy1_matched - xy0_proj_matched, dim=-1)

    dist = torch.stack([dist_in_img0, dist_in_img1]).mean(0)

    def get_matches_color(dist_matched: Tensor) -> Tensor:
        color_matches = torch.tensor([1.0, 0.0, 0.0])[None].repeat(
            dist_matched.shape[0], 1
        )  # set everything as red
        color_matches[dist_matched < 3.0] = torch.tensor(
            [1.0, 0.5, 0.0]
        )  # orange for 2 <= error < 3
        color_matches[dist_matched < 2.0] = torch.tensor(
            [1.0, 1.0, 0.0]
        )  # yellow for 1 <= error < 2
        color_matches[dist_matched < 1.0] = torch.tensor(
            [0.0, 1.0, 0.0]
        )  # lime for error < 1
        return color_matches

    color = get_matches_color(dist)

    if show_all_keypoints:
        unmatched_mask0 = torch.ones(xy0.shape[0], dtype=torch.bool, device=xy0.device)
        unmatched_mask0[matches[:, 0]] = False
        unmatched_mask1 = torch.ones(xy1.shape[0], dtype=torch.bool, device=xy1.device)
        unmatched_mask1[matches[:, 1]] = False
        scatter(
            xy0[unmatched_mask0], ax=axes[0], radius=radius / 2, c="black", linewidth=1
        )
        scatter(
            xy1[unmatched_mask1], ax=axes[1], radius=radius / 2, c="black", linewidth=1
        )

    scatter(xy0_matched, ax=axes[0], radius=radius, c=color, linewidth=1)
    scatter(xy1_matched, ax=axes[1], radius=radius, c=color, linewidth=1)

    for i in range(matches.shape[0]):
        thickness = 2 if highlight_mask is None or not highlight_mask[i].item() else 3
        alpha = (
            matches_alpha
            if highlight_mask is None or not highlight_mask[i].item()
            else 1.0
        )

        con_patch = ConnectionPatch(
            xyA=xy0_matched[i].cpu().numpy(),
            xyB=xy1_matched[i].cpu().numpy(),
            coordsA="data",
            coordsB="data",
            axesA=axes[0],
            axesB=axes[1],
            color=color[i].cpu().numpy(),
            linewidth=thickness,
            alpha=alpha,
        )
        axes[1].add_artist(con_patch)

    plt.pause(0.01)
    return fig, axes


def plot_image_pair_with_keypoints(
    img0: Tensor | np.ndarray,
    img1: Tensor | np.ndarray,
    xy0: Tensor,
    xy1: Tensor,
    matches: Tensor | None = None,
    matches_color: Tensor | None = None,
    indexes0: Tensor | None = None,
    indexes1: Tensor | None = None,
    plot_matches: bool = True,
    radius: float = 1.0,
    cmap: str | None = None,
    thicknesses: Tensor | None = None,
    matches_alpha: float = 0.5,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: float | None = None,
    axes: np.ndarray = None,
    name: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Args:
        img0: image 0
        img1: image 1
        xy0: keypoints in image 0 (x, y) with convention top-left pixel center at coordinate (0.5, 0.5)
            n0, 2
        xy1: keypoints in image 1 (x, y) with convention top-left pixel center at coordinate (0.5, 0.5)
            n1, 2
        matches: matches formatted as (index of matched xy0, index of matched xy1)
            n_matches, 2
        matches_color:
            n_matches or n_matches,3
        indexes0: indexes of the keypoints in image 0. If provided, the indexes are plotted
        indexes1: indexes of the keypoints in image 1. If provided, the indexes are plotted
        plot_matches: if True, plot the matches
        radius: the radius of the scatter
        cmap: the colormap to use for points and matches
        thicknesses: if provided, the thickness of the lines is proportional to the confidence
        matches_alpha: the alpha of the matches
        title: the plot title
        dpi: the figure dpi
        figsize: the figure size
        axes: if provided, do not open a new figure but instead plot on the provided axes
        name: the name of the figure
    """
    assert img0.ndim in {2, 3} and img1.ndim in {2, 3}, "image.ndim must be 2 or 3"
    assert xy0.ndim == 2 and xy1.ndim == 2, "xy must be a 2D tensor"
    if matches is not None:
        assert matches.ndim == 2, "matches must be a 2D tensor"
    cmap_function = matplotlib.colormaps[cmap] if cmap is not None else None

    if axes is None:
        fig, axes = subplots(1, 2, figsize=figsize, dpi=dpi, name=name)
        fig.suptitle(title) if title is not None else None
    else:
        fig = axes[0].figure
        assert axes.shape == (2,)
    imshow(img0, ax=axes[0], show=False, cmap="gray")
    imshow(img1, ax=axes[1], show=False, cmap="gray")

    if matches is not None:
        valid_matches_indexes = matches[matches != -1].view(-1, 2)
        xy0_unmatched_mask = torch.ones(
            xy0.shape[0], dtype=torch.bool, device=xy0.device
        )
        xy0_unmatched_mask[valid_matches_indexes[:, 0]] = False
        xy1_unmatched_mask = torch.ones(
            xy1.shape[0], dtype=torch.bool, device=xy1.device
        )
        xy1_unmatched_mask[valid_matches_indexes[:, 1]] = False

        xy0_matched = xy0[valid_matches_indexes[:, 0]]
        xy1_matched = xy1[valid_matches_indexes[:, 1]]
        xy0_unmatched = xy0[xy0_unmatched_mask]
        xy1_unmatched = xy1[xy1_unmatched_mask]
        indexes0_matched = (
            indexes0[valid_matches_indexes[:, 0]] if indexes0 is not None else None
        )
        indexes1_matched = (
            indexes1[valid_matches_indexes[:, 1]] if indexes1 is not None else None
        )
        indexes0_unmatched = (
            indexes0[xy0_unmatched_mask] if indexes0 is not None else None
        )
        indexes1_unmatched = (
            indexes1[xy1_unmatched_mask] if indexes1 is not None else None
        )
        scatter(xy0_unmatched, ax=axes[0], texts=indexes0_unmatched, radius=radius)
        scatter(xy1_unmatched, ax=axes[1], texts=indexes1_unmatched, radius=radius)
        colors = []
        for i in range(xy0_matched.shape[0]):
            if matches_color is None:
                color = (0.0, 1.0, 0.0)
            elif matches_color.ndim == 1:
                color = (
                    cmap_function(matches_color[i].item())
                    if cmap is not None
                    else (0.0, 1.0, 0.0)
                )
            else:
                color = matches_color[i].cpu().numpy()

            colors += [color]
            if plot_matches:
                thickness = 1 if thicknesses is None else thicknesses[i].item()
                con_patch = ConnectionPatch(
                    xyA=xy0_matched[i].cpu().numpy(),
                    xyB=xy1_matched[i].cpu().numpy(),
                    coordsA="data",
                    coordsB="data",
                    axesA=axes[0],
                    axesB=axes[1],
                    color=color,
                    linewidth=thickness,
                    alpha=matches_alpha,
                )
                axes[1].add_artist(con_patch)
        colors = np.array(colors)

        scatter(
            xy0_matched,
            ax=axes[0],
            texts=indexes0_matched,
            c=torch.tensor(colors),
            radius=radius,
        )
        scatter(
            xy1_matched,
            ax=axes[1],
            texts=indexes1_matched,
            c=torch.tensor(colors),
            radius=radius,
        )
    else:
        scatter(xy0, ax=axes[0], texts=indexes0, radius=radius)
        scatter(xy1, ax=axes[1], texts=indexes1, radius=radius)
    return fig, axes


def matching_plot(
    img0: Tensor,
    img1: Tensor,
    xy0: Tensor,
    xy1: Tensor,
    matches: MatchesWithExtra,
    plot_matches: bool = True,
    title: str = None,
    figsize: tuple[int, int] | None = None,
    dpi: float | None = None,
    axes: np.ndarray | None = None,
    name: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Args:
        img0: image 0
        img1: image 1
        xy0: keypoints in image 0 (x, y) with convention top-left pixel center at coordinate (0.5, 0.5)
            n0, 2
        xy1: keypoints in image 1 (x, y) with convention top-left pixel center at coordinate (0.5, 0.5)
            n1, 2
        matches: matches formatted as (index of matched xy0, index of matched xy1)
            n_matches, 2
        plot_matches: if True, plot the matches
        title: the plot title
        figsize: the figure size
        dpi: the figure dpi
        axes: if provided, do not open a new figure but instead plot on the provided axes
        name: the name of the figure
    """
    assert img0.ndim == 2 or (
        img0.ndim == 3 and img0.shape[0] in {1, 3}
    ), "image.ndim must be 2 or 3"
    assert img1.ndim == 2 or (
        img1.ndim == 3 and img1.shape[0] in {1, 3}
    ), "image.ndim must be 2 or 3"
    assert (
        xy0.ndim == 2 and xy1.ndim == 2
    ), f"xy must be a 2D tensor, got {xy0.ndim} and {xy1.ndim}"
    assert matches.shape == (
        xy0.shape[0],
        xy1.shape[0],
    ), f"matches must be a 2D tensor of shape {xy0.shape[0], xy1.shape[0]}, got {matches.shape}"

    matches_unsure_idxs = matches.matching_matrix_extra.unsure.nonzero()
    matches_inexistent_idxs = matches.matching_matrix_extra.inexistent.nonzero()
    matches_mismatch_idxs = matches.matching_matrix_extra.mismatched.nonzero()
    matches_correct_idxs = matches.matching_matrix_extra.correct.nonzero()

    # set the matches rgb color
    matches_unsure_color = torch.tensor([0.0, 0.0, 1.0])  # blue
    matches_inexistent_color = torch.tensor([1.0, 0.0, 1.0])  # magenta
    matches_mismatch_color = torch.tensor([1.0, 0.8, 0.0])  # orange
    matches_correct_color = torch.tensor([0.2, 0.8, 0.2])  # lime

    matches = torch.cat(
        (
            matches_unsure_idxs,
            matches_inexistent_idxs,
            matches_mismatch_idxs,
            matches_correct_idxs,
        )
    )
    matches_color = torch.cat(
        (
            matches_unsure_color.repeat(matches_unsure_idxs.shape[0], 1),
            matches_inexistent_color.repeat(matches_inexistent_idxs.shape[0], 1),
            matches_mismatch_color.repeat(matches_mismatch_idxs.shape[0], 1),
            matches_correct_color.repeat(matches_correct_idxs.shape[0], 1),
        )
    )

    fig, axes = plot_image_pair_with_keypoints(
        img0,
        img1,
        xy0,
        xy1,
        matches=matches,
        matches_color=matches_color,
        plot_matches=plot_matches,
        title=title,
        axes=axes,
        figsize=figsize,
        dpi=dpi,
        name=name,
    )
    return fig, axes
