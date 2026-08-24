import numpy as np


def save_fig_png(fig, png_path: str, dpi: int = 300):
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')


def save_fig_pdf(fig, pdf_path, max_points=2000, dpi=300):
    """
    For each line in the figure: if it has more than max_points,
    replace data with a decimated version and mark it rasterized.
    Then save the figure to `pdf_path` at `dpi`.
    """
    for ax in fig.axes:
        for line in ax.get_lines():
            x, y = line.get_data()
            if len(x) > max_points:
                # x2, y2 = decimate_slow(x, y, max_points)
                x2, y2 = decimate_fast(x, y, max_points)
                line.set_data(x2, y2)
                line.set_rasterized(True)
        # also rasterize large collections (e.g. scatter)
        for coll in ax.collections:
            # get approximate number of points (some collections expose offsets)
            offsets = getattr(coll, "get_offsets", lambda: None)()
            if offsets is not None and len(offsets) > max_points:
                coll.set_offsets(offsets[::max(1, len(offsets) // max_points)])
                coll.set_rasterized(True)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')


def decimate_fast(x, y, max_points, method="index"):
    """
    Uniformly decimate (x, y) to exactly max_points.
    - method="index": pick indices uniformly across the input index range (fast).
    - method="x": partition the x-range into max_points bins and pick one representative per bin
                  (tries to ensure even coverage in x).
    Endpoints are preserved when possible.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n <= max_points:
        return x, y

    if method == "index":
        idx = np.linspace(0, n - 1, max_points).round().astype(int)
        idx[0] = 0
        idx[-1] = n - 1
        idx = np.unique(idx)  # guard against rounding duplicates
        # if uniqueness reduced count, fill from remaining indices
        if idx.size < max_points:
            remaining = np.setdiff1d(np.arange(n), idx)
            need = max_points - idx.size
            extra = remaining[np.linspace(0, len(remaining) - 1, need).round().astype(int)]
            idx = np.sort(np.concatenate((idx, extra)))
        return x[idx], y[idx]

    elif method == "x":
        bins = np.linspace(x.min(), x.max(), max_points + 1)
        selected = []
        for b in range(len(bins) - 1):
            left, right = bins[b], bins[b + 1]
            if b == len(bins) - 2:
                in_bin = np.where((x >= left) & (x <= right))[0]
            else:
                in_bin = np.where((x >= left) & (x < right))[0]
            if in_bin.size == 0:
                center = 0.5 * (left + right)
                chosen = int(np.argmin(np.abs(x - center)))
            else:
                chosen = in_bin[len(in_bin) // 2]  # median-like pick to avoid extremes
            selected.append(chosen)
            if len(selected) >= max_points:
                break
        idx = np.unique(np.array(selected, dtype=int))
        if idx.size < max_points:
            remaining = np.setdiff1d(np.arange(n), idx)
            need = max_points - idx.size
            extra = remaining[np.linspace(0, len(remaining) - 1, need).round().astype(int)]
            idx = np.sort(np.concatenate((idx, extra)))
        else:
            idx = idx[:max_points]
        return x[idx], y[idx]

    else:
        raise ValueError("method must be 'index' or 'x'")


def decimate_slow(x, y, max_points):
    """
    Decimate (x, y) to at most max_points while preserving shape and avoiding empty regions.
    - First apply Ramer-Douglas-Peucker (RDP) with a binary search on epsilon to get <= max_points.
    - Then ensure coverage by filling empty x-bins with nearest original points until max_points reached.
    Endpoints are always kept.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n <= max_points:
        return x, y

    pts = np.column_stack((x, y))

    # RDP implementation returning indices kept (sorted)
    def _perp_dist(pt, a, b):
        # perpendicular distance from pt to segment a-b
        if np.allclose(a, b):
            return np.linalg.norm(pt - a)
        # area-based distance
        return np.abs(np.cross(b - a, a - pt)) / np.linalg.norm(b - a)

    def _rdp_indices(points):
        stack = [(0, len(points) - 1)]
        keep = {0, len(points) - 1}
        while stack:
            i, j = stack.pop()
            a, b = points[i], points[j]
            if j <= i + 1:
                continue
            sub = points[i + 1:j]
            if sub.size == 0:
                continue
            dists = np.apply_along_axis(_perp_dist, 1, sub, a, b)
            idx = np.argmax(dists)
            maxd = dists[idx]
            if maxd > _rdp_eps:
                k = i + 1 + idx
                keep.add(k)
                stack.append((i, k))
                stack.append((k, j))
        return np.array(sorted(keep), dtype=int)

    # Binary search epsilon to get no more than max_points (if possible)
    # initial bounds:
    diag = np.hypot(np.ptp(x), np.ptp(y))
    lo = 0.0
    hi = diag if diag > 0 else 1.0
    best_idx = np.arange(n)  # fallback: keep all
    global _rdp_eps  # used inside _rdp_indices
    for _ in range(40):
        _rdp_eps = (lo + hi) / 2
        idx = _rdp_indices(pts)
        m = len(idx)
        if m > max_points:
            lo = _rdp_eps  # need larger epsilon
        else:
            best_idx = idx
            hi = _rdp_eps  # try smaller epsilon to keep more detail
        if hi - lo < 1e-12:
            break

    kept = list(best_idx)
    kept_set = set(kept)

    # If still too many (unlikely), uniformly subsample kept indices (keep endpoints)
    if len(kept) > max_points:
        # ensure first and last preserved
        mid = kept[1:-1]
        take = max_points - 2
        if take <= 0:
            kept = [kept[0], kept[-1]]
        else:
            pick = np.linspace(0, len(mid) - 1, take).round().astype(int)
            kept = [kept[0]] + [mid[i] for i in pick] + [kept[-1]]
        kept = sorted(set(kept))

    # Ensure coverage: no empty bins across x-range. Create max_points bins and add a representative if empty.
    # Build a boolean mask for selected original indices
    selected = np.zeros(n, dtype=bool)
    selected[kept] = True

    if len(kept) < max_points:
        # bins across x-range (use max_points bins to aim for one per bin)
        bins = np.linspace(x.min(), x.max(), max_points + 1)
        # for each bin check if selected point present
        to_add = []
        for b in range(len(bins) - 1):
            left, right = bins[b], bins[b + 1]
            # find any selected inside bin
            in_bin_selected = np.where(selected & (x >= left) & (x < right))[0]
            if in_bin_selected.size > 0:
                continue
            # find original points in bin; if none, find closest to bin center
            in_bin_all = np.where((x >= left) & (x < right))[0]
            if in_bin_all.size > 0:
                # pick the point with largest absolute y-change inside bin (or just midpoint)
                if in_bin_all.size == 1:
                    chosen = in_bin_all[0]
                else:
                    # prefer the point with maximum local variation to preserve features
                    diffs = np.zeros(in_bin_all.size)
                    for ii, idx in enumerate(in_bin_all):
                        if idx == 0:
                            diffs[ii] = abs(y[idx + 1] - y[idx])
                        elif idx == n - 1:
                            diffs[ii] = abs(y[idx] - y[idx - 1])
                        else:
                            diffs[ii] = abs(y[idx + 1] - y[idx - 1]) / 2.0
                    chosen = in_bin_all[np.argmax(diffs)]
            else:
                # no original in bin (rare due to bins choice) -> pick closest by x to bin center
                center = 0.5 * (left + right)
                chosen = int(np.argmin(np.abs(x - center)))
            if not selected[chosen]:
                to_add.append(chosen)
                selected[chosen] = True
            if np.count_nonzero(selected) >= max_points:
                break

        # add selected indices while respecting max_points
        final_idx = np.where(selected)[0].tolist()
        if len(final_idx) > max_points:
            # uniformly drop extras (but keep endpoints)
            final_idx = sorted(final_idx)
            keep_front = final_idx[0]
            keep_back = final_idx[-1]
            middle = final_idx[1:-1]
            take = max_points - 2
            if take <= 0:
                final_idx = [keep_front, keep_back]
            else:
                pick = np.linspace(0, len(middle) - 1, take).round().astype(int)
                final_idx = [keep_front] + [middle[i] for i in pick] + [keep_back]
        else:
            final_idx = sorted(final_idx)
    else:
        final_idx = kept

    final_idx = np.array(sorted(set(final_idx)), dtype=int)
    # Final safety: if still more than max_points (edge cases), uniformly sample while keeping endpoints
    if final_idx.size > max_points:
        front, back = final_idx[0], final_idx[-1]
        middle = final_idx[1:-1]
        take = max_points - 2
        if take <= 0:
            final_idx = np.array([front, back], dtype=int)
        else:
            pick = np.linspace(0, len(middle) - 1, take).round().astype(int)
            final_idx = np.array([front] + [middle[i] for i in pick] + [back], dtype=int)

    return x[final_idx], y[final_idx]
