import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon
from skimage.graph import route_through_array
from skimage.draw import line as skline
import matplotlib.pyplot as plt


### resample each mapped fracture to evenly spaced vertices
def resample_geometry(geom, step):
    """Resample vertices at a fixed `step` spacing (identical across all features).
    Spacing is exactly `step` everywhere; the final vertex is the line endpoint,
    so only the last interval may be shorter than step.
    """
    if geom is None or geom.is_empty:
        return geom

    def resample_linestring(ls):
        if ls.length == 0:
            return ls
        ds = np.arange(0.0, ls.length, step)          # exact `step` spacing
        ds = np.append(ds, ls.length)                 # keep the true endpoint
        return LineString([ls.interpolate(d) for d in ds])

    if geom.geom_type == "LineString":
        return resample_linestring(geom)
    if geom.geom_type == "MultiLineString":
        return MultiLineString([resample_linestring(ls) for ls in geom.geoms])
    return geom


####### Generate reference fault trace based on Thomas et al. (2023)
# code modified from Python version of Thomas et al. (2023)'s code https://zenodo.org/records/11175147

def smooth_lcp(lcp, tolerance=50.0, n_out=None):
    """Smooth a line with a PAEK-like arc-length kernel, akin to the ArcGIS Smooth Line
    tool. Each output point is a Gaussian-weighted average of nearby vertices in
    arc-length, removing grid-scale staircase artifacts while preserving the
    longer-wavelength bends of the trace. Endpoints are fixed.
    Inputs:
    - lcp: shp of line
    - tolerance: smoothing length (m) to smooth out raster pixel knicks, larger = smoother 
    - n_out: number of output vertices (default: keep the input count)
    """
    line = lcp.geometry.iloc[0] if isinstance(lcp, gpd.GeoDataFrame) else lcp
    xy = np.asarray(line.coords)[:, :2]
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]   # cumulative arc length
    L = seg[-1]
    s_out = seg if n_out is None else np.linspace(0.0, L, n_out)
    sigma = float(tolerance)
    half = 4.0 * sigma                                             # kernel support
    out = np.empty((len(s_out), 2))
    for i, s in enumerate(s_out):
        lo = np.searchsorted(seg, s - half, "left")
        hi = np.searchsorted(seg, s + half, "right")
        w = np.exp(-0.5 * ((seg[lo:hi] - s) / sigma) ** 2)
        out[i] = (w @ xy[lo:hi]) / w.sum()
    out[0], out[-1] = xy[0], xy[-1]                                # keep endpoints fixed
    geom = LineString(out)
    return gpd.GeoDataFrame(geometry=[geom], crs=lcp.crs) if isinstance(lcp, gpd.GeoDataFrame) else geom


def generate_lcp(shp_path, epsg, start_coords, end_coords, res=10.0, smooth_tol=50.0):
    """
    Produce lcp line from shapefile following approach in Thomas et al. (2023).
    Inputs:
    - Shapefile
    - Coordinate system
    - Start and end points for rupture trace
    - res: cost grid resolution (m)
    - smooth_tol: PAEK-like smoothing length (m) applied to the raster
    the grid staircase; set 0 or None to return the raw path.
    """
    fault_cost, nonfault_cost = 1.0, 100.0  # fixed from defaults in Thomas et al. (2023)

    gdf = gpd.read_file(shp_path).to_crs(epsg=epsg)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin -= res; ymin -= res; xmax += res; ymax += res
    ncols = int(np.ceil((xmax - xmin) / res))
    nrows = int(np.ceil((ymax - ymin) / res))

    def xy_to_rc(x, y):
        c = min(max(int((x - xmin) / res), 0), ncols - 1)
        r = min(max(int((ymax - y) / res), 0), nrows - 1)
        return r, c

    def rc_to_xy(r, c):
        return xmin + (c + 0.5) * res, ymax - (r + 0.5) * res

    cost = np.full((nrows, ncols), nonfault_cost, dtype=float)  # burn every line segment into the cost grid

    for geom in gdf.geometry:
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for ln in lines:
            rc = [xy_to_rc(x, y) for x, y in np.asarray(ln.coords)[:, :2]]
            for (r0, c0), (r1, c1) in zip(rc[:-1], rc[1:]):
                rr, cc = skline(r0, c0, r1, c1)
                cost[rr, cc] = fault_cost

    start = xy_to_rc(start_coords.x, start_coords.y)
    end   = xy_to_rc(end_coords.x, end_coords.y)

    idx, _ = route_through_array(cost, start, end,
                                geometric=True, fully_connected=True)
    path_xy = np.array([rc_to_xy(r, c) for r, c in idx])

    geom = LineString(path_xy)
    if smooth_tol:
        geom = smooth_lcp(geom, tolerance=smooth_tol)   # remove grid staircase
        path_xy = np.asarray(geom.coords)

    lcp = gpd.GeoDataFrame(geometry=[geom], crs=f"EPSG:{epsg}")
    return lcp, path_xy

### oriented rectangular window aligned to the local LCP strike
def oriented_window(line, d, L, along_half, across_half):
    """Oriented rectangle centered on the LCP at distance d, aligned to the local
    strike. Half-length along_half along strike, half-width across_half across strike.
    Returns the center, the along/across unit vectors, and the rectangle polygon.
    """
    c = line.interpolate(min(d, L))
    c0 = np.array([c.x, c.y])
    a   = line.interpolate(max(d - along_half, 0.0))
    bpt = line.interpolate(min(d + along_half, L))
    along_dir = np.array([bpt.x - a.x, bpt.y - a.y])
    nrm = np.linalg.norm(along_dir)
    along_dir = along_dir / nrm if nrm else np.array([1.0, 0.0])
    perp_dir = np.array([-along_dir[1], along_dir[0]])
    corners = [c0 + sa * along_half * along_dir + sp * across_half * perp_dir
            for sa, sp in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return c0, along_dir, perp_dir, Polygon(corners)


######### Measure number of strands and fault roughness based on ruptures inside a window
def roughness_n_strands_along_lcp(lcp, rupture_map, along_half, across_half, along_trace_step):
    """Walk the LCP, window the rupture map with an oriented rectangle at each step, and
    measure strand count and normalized RMS roughness per window.
    Inputs:
    - LCP line
    - Rupture map
    - Half-length along strike (along_half)
    - Half-width across strike (across_half)
    - Stepping distance along rupture (along_trace_step)
    """
    line = lcp.geometry.iloc[0] if isinstance(lcp, gpd.GeoDataFrame) else lcp
    L = line.length
    dists = np.arange(0.0, L + along_trace_step, along_trace_step)

    rows = []
    for d in dists:
        c0, along_dir, perp_dir, window = oriented_window(line, d, L, along_half, across_half)
        cx, cy = c0
        hits = rupture_map[rupture_map.intersects(window)]  # n of strands inside or intercepting window
        n_strands = len(hits)

        # empty window -> 0 strands and nan roughness
        if n_strands == 0:
            rows.append((d, cx, cy, np.nan, np.nan))
            continue

        # collect vertices of the strands clipped to the window for roughness measurement
        pts = []
        for g in hits.geometry.intersection(window):
            if g.is_empty:
                continue
            parts = g.geoms if g.geom_type in ("MultiLineString", "GeometryCollection") else [g]
            for p in parts:
                if p.geom_type == "LineString":
                    pts.extend(np.asarray(p.coords)[:, :2])
        pts = np.asarray(pts)
        if len(pts) < 2:
            rows.append((d, cx, cy, n_strands, np.nan))
            continue

        # total-least-squares straight-line fit, perpendicular RMS, normalize by fit length
        cen = pts - pts.mean(axis=0)
        _, _, vh = np.linalg.svd(cen, full_matrices=False)
        fit_dir = vh[0]
        fit_perp = np.array([-fit_dir[1], fit_dir[0]])
        perp = cen @ fit_perp
        diag = np.hypot(2 * along_half, 2 * across_half)
        chord = LineString([pts.mean(axis=0) - diag * fit_dir,
                            pts.mean(axis=0) + diag * fit_dir]).intersection(window)
        straight_len = chord.length
        roughness = np.sqrt(np.mean(perp ** 2)) / straight_len if straight_len > 0 else np.nan
        rows.append((d, cx, cy, n_strands, roughness))

    return pd.DataFrame(rows, columns=["distance along lcp line", "x_coord_midpoint",
                                    "y_coord_midpoint", "n_strands", "roughness"])


######### Measure rupture zone width orthogonal to the LCP
def rupture_zone_width_along_lcp(lcp, rupture_map, along_half, across_half, along_trace_step, width_bin=50.0):
    """Walk the LCP, window with an oriented rectangle at each step, and measure rupture zone
    width orthogonal to the local LCP strike: bin clipped vertices along strike, take the span
    between the furthest-out crack on either side per bin, and report the max over bins.
    Inputs:
    - LCP line
    - Rupture map
    - Half-length along strike (along_half)
    - Half-width across strike (across_half)
    - Stepping distance along rupture (along_trace_step)
    - Along-strike bin size (width_bin)
    """
    line = lcp.geometry.iloc[0] if isinstance(lcp, gpd.GeoDataFrame) else lcp
    L = line.length
    dists = np.arange(0.0, L + along_trace_step, along_trace_step)

    rows = []
    for d in dists:
        c0, along_dir, perp_dir, window = oriented_window(line, d, L, along_half, across_half)
        cx, cy = c0
        hits = rupture_map[rupture_map.intersects(window)]

        if len(hits) == 0:
            rows.append((d, cx, cy, np.nan))
            continue

        pts = []
        for g in hits.geometry.intersection(window):
            if g.is_empty:
                continue
            parts = g.geoms if g.geom_type in ("MultiLineString", "GeometryCollection") else [g]
            for p in parts:
                if p.geom_type == "LineString":
                    pts.extend(np.asarray(p.coords)[:, :2])
        pts = np.asarray(pts)
        if len(pts) < 2:
            rows.append((d, cx, cy, np.nan))
            continue

        cen = pts - c0          # offsets from the LCP point
        s = cen @ along_dir
        perp = cen @ perp_dir

        nb = max(int(np.ceil((s.max() - s.min()) / width_bin)), 1)
        edges = np.linspace(s.min(), s.max(), nb + 1)
        b = np.clip(np.digitize(s, edges) - 1, 0, nb - 1)
        width = 0.0
        for j in range(nb):
            pj = perp[b == j]
            if pj.size:
                width = max(width, pj.max() - pj.min())

        rows.append((d, cx, cy, width))

    return pd.DataFrame(rows, columns=["distance along lcp line", "x_coord_midpoint",
                                    "y_coord_midpoint", "rupture_zone_width"])


### debugging block: roughness and number of strands
def plot_roughness_windows(lcp, rupture_map, df, along_half, across_half, n=6, random_state=None):
    """Plot n randomly sampled oriented-rectangle windows. Each rupture line is drawn in its own
    color, with the fit line and the perpendicular distance from each vertex; n strands and
    roughness are printed on top of each window.
    Inputs:
    - LCP line
    - Rupture map
    - Dataframe with roughness and n_strands measurements
    - Half-length along strike (along_half)
    - Half-width across strike (across_half)
    """
    line = lcp.geometry.iloc[0] if isinstance(lcp, gpd.GeoDataFrame) else lcp
    L = line.length

    valid = df.dropna(subset=["roughness"])
    if valid.empty:
        print("No windows with a roughness value to plot.")
        return
    n = min(n, len(valid))
    sample = valid.sample(n, random_state=random_state).sort_values("distance along lcp line")

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols)
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        d = row["distance along lcp line"]
        c0, along_dir, perp_dir, window = oriented_window(line, d, L, along_half, across_half)
        hits = rupture_map[rupture_map.intersects(window)]

        # one color per map feature, so the number of distinct colors == n_strands
        n_feat = len(hits)
        cmap = plt.get_cmap("tab20" if n_feat <= 20 else "gist_ncar", max(n_feat, 1))

        pts = []
        for k, g in enumerate(hits.geometry.intersection(window)):
            if g.is_empty:
                continue
            geoms = g.geoms if g.geom_type in ("MultiLineString", "GeometryCollection") else [g]
            for p in geoms:
                if p.geom_type == "LineString":
                    xy = np.asarray(p.coords)[:, :2]
                    ax.plot(xy[:, 0], xy[:, 1], color=cmap(k), lw=1.0, zorder=1)
                    ax.scatter(xy[:, 0], xy[:, 1], s=10, color=cmap(k), zorder=5)
                    pts.extend(xy)
        pts = np.asarray(pts)

        px, py = window.exterior.xy
        ax.plot(px, py, color="lightgray", lw=1.2, zorder=3)

        if len(pts) >= 2:
            mean = pts.mean(axis=0)
            cen = pts - mean
            _, _, vh = np.linalg.svd(cen, full_matrices=False)
            fit_dir = vh[0]
            s = cen @ fit_dir
            feet = mean + np.outer(s, fit_dir)
            diag = np.hypot(2 * along_half, 2 * across_half)
            chord = LineString([mean - diag * fit_dir,
                                mean + diag * fit_dir]).intersection(window)
            cxy = np.asarray(chord.coords)
            ax.plot(cxy[:, 0], cxy[:, 1], color="tab:red", lw=1, zorder=1)
            for pt, ft in zip(pts, feet):
                ax.plot([pt[0], ft[0]], [pt[1], ft[1]], color="dimgray", lw=0.6, zorder=2)

        ax.text(0.04, 0.96, f"n={int(row['n_strands'])}\n Roughness={row['roughness']:.4f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{d:.0f} m along LCP", fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()


### debugging block: rupture zone width
def plot_rupture_zone_width_windows(lcp, rupture_map, df, along_half, across_half, n=6,
                                    width_bin=50.0, random_state=None):
    """Plot n randomly sampled oriented-rectangle windows for rupture zone width. Each rupture
    line is drawn in its own color, the LCP reference line is shown, the across-strike envelope
    (furthest-out cracks per along-strike bin) is shaded, the bin setting the max width is
    marked, and n strands and RZW are printed.
    Inputs:
    - LCP line
    - Rupture map
    - Dataframe with rupture_zone_width (and n_strands) measurements
    - Half-length along strike (along_half)
    - Half-width across strike (across_half)
    - Along-strike bin size (width_bin, must match the measurement call)
    """
    line = lcp.geometry.iloc[0] if isinstance(lcp, gpd.GeoDataFrame) else lcp
    L = line.length

    valid = df[df["rupture_zone_width"] > 0]
    if valid.empty:
        print("No windows with a rupture zone width to plot.")
        return
    n = min(n, len(valid))
    sample = valid.sample(n, random_state=random_state).sort_values("distance along lcp line")

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols)
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        d = row["distance along lcp line"]
        c0, along_dir, perp_dir, window = oriented_window(line, d, L, along_half, across_half)
        hits = rupture_map[rupture_map.intersects(window)]

        n_feat = len(hits)
        cmap = plt.get_cmap("tab20" if n_feat <= 20 else "gist_ncar", max(n_feat, 1))

        pts = []
        for k, g in enumerate(hits.geometry.intersection(window)):
            if g.is_empty:
                continue
            geoms = g.geoms if g.geom_type in ("MultiLineString", "GeometryCollection") else [g]
            for p in geoms:
                if p.geom_type == "LineString":
                    xy = np.asarray(p.coords)[:, :2]
                    ax.plot(xy[:, 0], xy[:, 1], color=cmap(k), lw=1.0, zorder=1)
                    ax.scatter(xy[:, 0], xy[:, 1], s=10, color=cmap(k), zorder=5)
                    pts.extend(xy)
        pts = np.asarray(pts)

        px, py = window.exterior.xy
        ax.plot(px, py, color="lightgray", lw=1.2, zorder=3)

        # LCP reference line clipped to the window
        ref = line.intersection(window)
        if not ref.is_empty:
            rparts = ref.geoms if ref.geom_type in ("MultiLineString", "GeometryCollection") else [ref]
            for rp in rparts:
                if rp.geom_type == "LineString":
                    rxy = np.asarray(rp.coords)[:, :2]
                    ax.plot(rxy[:, 0], rxy[:, 1], color="k", lw=1.5, zorder=4)

        if len(pts) >= 2:
            cen = pts - c0
            s = cen @ along_dir
            perp = cen @ perp_dir

            nb = max(int(np.ceil((s.max() - s.min()) / width_bin)), 1)
            edges = np.linspace(s.min(), s.max(), nb + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            b = np.clip(np.digitize(s, edges) - 1, 0, nb - 1)

            sc, hi, lo = [], [], []
            for j in range(nb):
                pj = perp[b == j]
                if pj.size:
                    sc.append(centers[j]); hi.append(pj.max()); lo.append(pj.min())
            sc = np.asarray(sc); hi = np.asarray(hi); lo = np.asarray(lo)

            def to_xy(svals, pvals):
                return c0 + np.outer(svals, along_dir) + np.outer(pvals, perp_dir)

            up_xy = to_xy(sc, hi)
            lo_xy = to_xy(sc, lo)
            band = np.vstack([up_xy, lo_xy[::-1]])
            ax.fill(band[:, 0], band[:, 1], color="0.5", alpha=0.15, zorder=1)

            jmax = int(np.argmax(hi - lo))
            tie = to_xy(np.array([sc[jmax], sc[jmax]]), np.array([lo[jmax], hi[jmax]]))
            ax.plot(tie[:, 0], tie[:, 1], color="crimson", lw=1.5, zorder=6)

        ax.text(0.04, 0.96, f"n={int(row['n_strands'])}\n RZW={row['rupture_zone_width']:.0f} m",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{d:.0f} m along LCP", fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
