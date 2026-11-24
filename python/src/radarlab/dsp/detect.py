"""Detection post-processing: clustering, peak grouping, sub-bin interpolation.

Clustering works on CFAR masks in (range bin, Doppler bin) coordinates with a
circular Doppler axis. Each cluster is represented by its maximum-power cell,
refined to fractional bins with a 3-point parabolic fit on log-power
(equivalently a Gaussian fit on magnitude), which has small bias for the Hann
main lobe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree


def parabolic_offset(ym: np.ndarray | float, y0: np.ndarray | float,
                     yp: np.ndarray | float) -> np.ndarray:
    """Vertex offset δ ∈ [−½, ½] of the parabola through (−1, ym), (0, y0), (1, yp)::

        δ = ½·(ym − yp) / (ym − 2·y0 + yp)
    """
    ym, y0, yp = (np.asarray(v, float) for v in (ym, y0, yp))
    den = ym - 2 * y0 + yp
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.abs(den) > 1e-300, 0.5 * (ym - yp) / den, 0.0)
    return np.clip(d, -0.5, 0.5)


def refine_peak(P: np.ndarray, r: np.ndarray, d: np.ndarray, log: bool = True,
                circular_doppler: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (range, Doppler) bins of peaks at integer cells (r, d) of a
    power map P[r, d] via separable parabolic interpolation on log-power.
    Range edges are not interpolated (offset 0)."""
    r = np.atleast_1d(np.asarray(r, int))
    d = np.atleast_1d(np.asarray(d, int))
    Nr, Md = P.shape
    L = np.log(np.maximum(P, 1e-300)) if log else P
    rm, rp = np.clip(r - 1, 0, Nr - 1), np.clip(r + 1, 0, Nr - 1)
    dr = parabolic_offset(L[rm, d], L[r, d], L[rp, d])
    dr = np.where((r == 0) | (r == Nr - 1), 0.0, dr)
    if circular_doppler:
        dm, dp = (d - 1) % Md, (d + 1) % Md
        dd = parabolic_offset(L[r, dm], L[r, d], L[r, dp])
    else:
        dm, dp = np.clip(d - 1, 0, Md - 1), np.clip(d + 1, 0, Md - 1)
        dd = np.where((d == 0) | (d == Md - 1), 0.0, parabolic_offset(L[r, dm], L[r, d], L[r, dp]))
    return r + dr, d + dd


def refine_peak_1d(y: np.ndarray, i: int, log: bool = True, circular: bool = False) -> float:
    """Fractional index of the peak at integer index i of a 1-D power sequence."""
    n = len(y)
    L = np.log(np.maximum(y, 1e-300)) if log else y
    if circular:
        return float(i + parabolic_offset(L[(i - 1) % n], L[i], L[(i + 1) % n]))
    if i <= 0 or i >= n - 1:
        return float(i)
    return float(i + parabolic_offset(L[i - 1], L[i], L[i + 1]))


@dataclass
class Cluster:
    """A group of CFAR cells. ``range_bin``/``doppler_bin`` are the integer
    coordinates of the strongest cell; ``cells`` is an (n, 2) array."""

    range_bin: int
    doppler_bin: int
    power: float
    size: int
    cells: np.ndarray


def _clusters_from_labels(P: np.ndarray, cells: np.ndarray, labels: np.ndarray) -> list[Cluster]:
    out: list[Cluster] = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        c = cells[labels == lab]
        pw = P[c[:, 0], c[:, 1]]
        i = int(np.argmax(pw))
        out.append(Cluster(int(c[i, 0]), int(c[i, 1]), float(pw[i]), len(c), c))
    out.sort(key=lambda cl: -cl.power)
    return out


def cluster_connected(mask: np.ndarray, P: np.ndarray, circular_doppler: bool = True) -> list[Cluster]:
    """8-connected components of the CFAR mask (Doppler wrap-around merged)."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return []
    if circular_doppler and mask.shape[1] > 2:
        # union-find merge of labels touching across the Doppler seam
        parent = np.arange(n + 1)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        L, R = lab[:, 0], lab[:, -1]
        Nr = mask.shape[0]
        for r in range(Nr):
            if L[r] == 0:
                continue
            for rr in (r - 1, r, r + 1):
                if 0 <= rr < Nr and R[rr] != 0:
                    a, b = find(L[r]), find(R[rr])
                    if a != b:
                        parent[b] = a
        lab = np.vectorize(find)(lab) * (lab > 0)
    cells = np.argwhere(lab > 0)
    return _clusters_from_labels(P, cells, lab[cells[:, 0], cells[:, 1]])


def cluster_dbscan(mask: np.ndarray, P: np.ndarray, eps: float = 1.5, min_samples: int = 1,
                   circular_doppler: bool = True) -> list[Cluster]:
    """DBSCAN (Ester et al., 1996) on CFAR cell coordinates (Euclidean in bins,
    periodic in Doppler). Noise points (label −1) are discarded."""
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return []
    Nr, Md = mask.shape
    box = [4 * Nr + 10, Md] if circular_doppler else None
    tree = cKDTree(cells.astype(float), boxsize=box)
    nbrs = tree.query_ball_point(cells.astype(float), eps)
    core = np.array([len(n) >= min_samples for n in nbrs])
    labels = -np.ones(len(cells), int)
    cid = 0
    for i in range(len(cells)):
        if labels[i] != -1 or not core[i]:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for q in nbrs[j]:
                if labels[q] == -1:
                    labels[q] = cid
                    stack.append(q)
        cid += 1
    return _clusters_from_labels(P, cells, labels)


def local_maxima(P: np.ndarray, mask: np.ndarray | None = None, circular_doppler: bool = True) -> np.ndarray:
    """Cells that are ≥ all 8 neighbours (Doppler wrap). Returns (n, 2) indices."""
    mode = ("nearest", "wrap") if circular_doppler else "nearest"
    mx = ndi.maximum_filter(P, size=3, mode=mode)
    pk = P >= mx
    if mask is not None:
        pk &= mask
    return np.argwhere(pk)
