"""Render shipping snapshots as interactive route maps.

The script expects coordinate-enriched snapshot CSVs created via
``scripts/create_snapshots.py`` (columns: ``u``, ``v``, ``w``,
``u_lat``, ``u_lon``, ``v_lat``, ``v_lon``). For each snapshot it aggregates
duplicate voyages, weights line thickness by both voyage frequency and cargo
volume, and produces an HTML map with Plotly ``Scattergeo`` layers. When
``result_kmeans_snapshot_*.csv`` anomaly outputs are available, the top X% of
routes (default 1%) are highlighted with dedicated lines, port markers, and
labels so analysts can quickly see the most anomalous connections.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DIR = ROOT / "data" / "snapshots"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "maps"
DEFAULT_ANOMALY_DIR = ROOT / "output" / "anomalies"
REQUIRED_COLUMNS = {"u", "v", "w", "u_lat", "u_lon", "v_lat", "v_lon"}


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-snapshot route maps")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR, help="Directory containing snapshot CSVs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store HTML maps")
    parser.add_argument("--count-weight", type=float, default=0.5, help="Relative weight for voyage frequency when sizing lines")
    parser.add_argument("--tonnage-weight", type=float, default=0.5, help="Relative weight for cargo tonnage when sizing lines")
    parser.add_argument("--max-routes", type=int, default=4000, help="Optional cap on number of thickest routes to draw (0 = all)")
    parser.add_argument("--min-score-quantile", type=float, default=0.0, help="Drop routes below this score quantile (0-1)")
    parser.add_argument("--min-line-width", type=float, default=0.8, help="Minimum line width in pixels")
    parser.add_argument("--max-line-width", type=float, default=6.5, help="Maximum line width in pixels")
    parser.add_argument("--line-opacity", type=float, default=0.35, help="Opacity applied to route segments")
    parser.add_argument("--line-colorscale", type=str, default="Turbo", help="Plotly colorscale name for routes")
    parser.add_argument("--min-marker-size", type=float, default=4.0, help="Smallest port marker size")
    parser.add_argument("--max-marker-size", type=float, default=16.0, help="Largest port marker size")
    parser.add_argument("--projection", type=str, default="natural earth", help="Plotly geo projection name (e.g. 'orthographic')")
    parser.add_argument("--anomaly-dir", type=Path, default=DEFAULT_ANOMALY_DIR, help="Directory containing result_kmeans_snapshot CSVs")
    parser.add_argument("--anomaly-prefix", type=str, default="result_kmeans_snapshot_", help="Prefix for anomaly CSV filenames")
    parser.add_argument("--highlight-top-n", type=int, default=10, help="Number of highest anomaly scores to highlight (0 disables)")
    parser.add_argument("--highlight-line-width", type=float, default=8.0, help="Line width for highlighted anomaly routes")
    parser.add_argument("--highlight-line-color", type=str, default="#d62728", help="Color used for highlighted anomaly routes")
    parser.add_argument("--highlight-port-size", type=float, default=11.0, help="Marker size for highlighted ports")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def list_snapshots(snapshot_dir: Path) -> list[Path]:
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")
    paths = sorted(snapshot_dir.glob("snapshot_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No snapshot_*.csv files found in {snapshot_dir}")
    return paths


def load_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    return df


def snapshot_period_from_stem(stem: str) -> str:
    return stem.replace("snapshot_", "", 1)


def load_top_anomalies(period_label: str, args: argparse.Namespace) -> pd.DataFrame:
    top_n = args.highlight_top_n
    if top_n <= 0:
        return pd.DataFrame(columns=["u", "v", "highlight_score", "highlight_rank"])

    anomaly_path = args.anomaly_dir / f"{args.anomaly_prefix}{period_label}.csv"
    if not anomaly_path.exists():
        logging.warning("Anomaly file not found for %s: %s", period_label, anomaly_path)
        return pd.DataFrame(columns=["u", "v", "highlight_score", "highlight_rank"])

    df = pd.read_csv(anomaly_path)
    required = {"u", "v"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{anomaly_path.name} missing columns: {sorted(missing)}")

    rank_col = "anomaly_score_delta_vs_prev" if "anomaly_score_delta_vs_prev" in df.columns else "anomaly_score"
    if rank_col not in df.columns:
        raise ValueError(
            f"{anomaly_path.name} missing ranking column: expected 'anomaly_score_delta_vs_prev' or 'anomaly_score'"
        )

    total_rows = len(df)
    if total_rows == 0:
        return pd.DataFrame(columns=["u", "v", "highlight_score", "highlight_rank"])

    top_n = min(top_n, total_rows)
    top = df.nlargest(top_n, rank_col).copy()
    top = top.rename(columns={rank_col: "highlight_score"})
    top["highlight_rank"] = range(1, len(top) + 1)
    keep_cols = [c for c in ["u", "v", "highlight_score", "highlight_rank", "is_disappeared"] if c in top.columns]
    top = top[keep_cols].copy()
    logging.debug("Loaded %s top anomalies for %s by %s", len(top), period_label, rank_col)
    return top


def build_route_geometry(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["u", "v", "u_lat", "u_lon", "v_lat", "v_lon"])
    geom = (
        df.groupby(["u", "v"], as_index=False)
        .agg(u_lat=("u_lat", "first"), u_lon=("u_lon", "first"), v_lat=("v_lat", "first"), v_lon=("v_lon", "first"))
    )
    return geom


def build_port_geometry(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["port", "lat", "lon"])

    origin = df[["u", "u_lat", "u_lon"]].rename(columns={"u": "port", "u_lat": "lat", "u_lon": "lon"})
    dest = df[["v", "v_lat", "v_lon"]].rename(columns={"v": "port", "v_lat": "lat", "v_lon": "lon"})
    ports = pd.concat([origin, dest], ignore_index=True)
    ports = ports.dropna(subset=["port", "lat", "lon"])
    return ports.drop_duplicates(subset=["port"], keep="first").reset_index(drop=True)


def enrich_highlight_geometry(
    highlights: pd.DataFrame,
    current_snapshot_df: pd.DataFrame,
    previous_snapshot_df: pd.DataFrame | None,
    period_label: str,
) -> pd.DataFrame:
    """Attach coordinates for strict top-N anomalies.

    Priority: current route geometry -> previous route geometry -> per-port lookup.
    """
    if highlights.empty:
        return highlights

    current_geom = build_route_geometry(current_snapshot_df)
    enriched = highlights.merge(current_geom, on=["u", "v"], how="left")
    enriched["highlight_source"] = "current"

    if previous_snapshot_df is not None and not previous_snapshot_df.empty:
        prev_geom = build_route_geometry(previous_snapshot_df)
        missing_mask = enriched[["u_lat", "u_lon", "v_lat", "v_lon"]].isna().any(axis=1)
        if missing_mask.any() and not prev_geom.empty:
            missing_idx = enriched.index[missing_mask]
            prev_fill = enriched.loc[missing_mask, ["u", "v"]].merge(prev_geom, on=["u", "v"], how="left")
            prev_fill.index = missing_idx
            for col in ["u_lat", "u_lon", "v_lat", "v_lon"]:
                enriched.loc[missing_mask, col] = enriched.loc[missing_mask, col].combine_first(prev_fill[col])
            filled_mask = missing_mask & enriched[["u_lat", "u_lon", "v_lat", "v_lon"]].notna().all(axis=1)
            enriched.loc[filled_mask, "highlight_source"] = "previous"

    combined_df = current_snapshot_df if previous_snapshot_df is None else pd.concat(
        [current_snapshot_df, previous_snapshot_df], ignore_index=True
    )
    port_geom = build_port_geometry(combined_df)

    missing_mask = enriched[["u_lat", "u_lon", "v_lat", "v_lon"]].isna().any(axis=1)
    if missing_mask.any() and not port_geom.empty:
        missing_idx = enriched.index[missing_mask]
        u_map = port_geom.rename(columns={"port": "u", "lat": "u_lat_port", "lon": "u_lon_port"})
        v_map = port_geom.rename(columns={"port": "v", "lat": "v_lat_port", "lon": "v_lon_port"})
        fill_df = (
            enriched.loc[missing_mask, ["u", "v"]]
            .merge(u_map, on="u", how="left")
            .merge(v_map, on="v", how="left")
        )
        fill_df.index = missing_idx
        enriched.loc[missing_mask, "u_lat"] = enriched.loc[missing_mask, "u_lat"].combine_first(fill_df["u_lat_port"])
        enriched.loc[missing_mask, "u_lon"] = enriched.loc[missing_mask, "u_lon"].combine_first(fill_df["u_lon_port"])
        enriched.loc[missing_mask, "v_lat"] = enriched.loc[missing_mask, "v_lat"].combine_first(fill_df["v_lat_port"])
        enriched.loc[missing_mask, "v_lon"] = enriched.loc[missing_mask, "v_lon"].combine_first(fill_df["v_lon_port"])

        filled_mask = missing_mask & enriched[["u_lat", "u_lon", "v_lat", "v_lon"]].notna().all(axis=1)
        enriched.loc[filled_mask, "highlight_source"] = "port_lookup"

    unresolved = enriched[["u_lat", "u_lon", "v_lat", "v_lon"]].isna().any(axis=1).sum()
    if unresolved > 0:
        logging.warning(
            "%s top anomalies in %s cannot be drawn due to missing coordinates",
            int(unresolved),
            period_label,
        )
        enriched = enriched.loc[~enriched[["u_lat", "u_lon", "v_lat", "v_lon"]].isna().any(axis=1)].copy()

    logging.info("Highlight routes for %s: requested=%s, drawable=%s", period_label, len(highlights), len(enriched))
    return enriched


def attach_highlight_metadata(routes: pd.DataFrame, highlights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if routes.empty or highlights.empty:
        routes = routes.copy()
        routes["highlight_score"] = pd.NA
        routes["highlight_rank"] = pd.NA
        return routes, routes.loc[routes["highlight_score"].notna()]

    merged = routes.merge(highlights, on=["u", "v"], how="left")
    highlight_routes = merged.loc[merged["highlight_score"].notna()].copy()
    return merged, highlight_routes


def aggregate_routes(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["u", "v", "u_lat", "u_lon", "v_lat", "v_lon"], as_index=False)
        .agg(voyage_count=("w", "count"), total_tonnage=("w", "sum"))
    )
    logging.debug("Aggregated %s unique routes", len(grouped))
    return grouped


def score_routes(routes: pd.DataFrame, count_weight: float, tonnage_weight: float) -> pd.DataFrame:
    if routes.empty:
        return routes
    routes = routes.copy()
    count_max = routes["voyage_count"].max() or 1.0
    tonnage_max = routes["total_tonnage"].max() or 1.0
    routes["count_score"] = routes["voyage_count"] / count_max
    routes["tonnage_score"] = routes["total_tonnage"] / tonnage_max
    weight_sum = max(count_weight + tonnage_weight, 1e-9)
    routes["score"] = (
        (routes["count_score"] * count_weight) + (routes["tonnage_score"] * tonnage_weight)
    ) / weight_sum
    return routes


def filter_and_rank_routes(routes: pd.DataFrame, min_score_q: float, max_routes: int) -> pd.DataFrame:
    if routes.empty:
        return routes
    if min_score_q > 0:
        threshold = routes["score"].quantile(min_score_q)
        routes = routes.loc[routes["score"] >= threshold]
    routes = routes.sort_values("score", ascending=False)
    if max_routes > 0:
        routes = routes.head(max_routes)
    routes.reset_index(drop=True, inplace=True)
    logging.debug("Keeping %s routes after filtering", len(routes))
    return routes


def assign_line_style(routes: pd.DataFrame, min_width: float, max_width: float, colorscale: str) -> pd.DataFrame:
    if routes.empty:
        return routes
    span = max_width - min_width
    routes = routes.copy()
    routes["line_width"] = min_width + routes["score"] * span
    colors = sample_colorscale(colorscale, routes["score"].clip(0, 1).tolist())
    routes["line_color"] = colors
    return routes


def compute_port_stats(routes: pd.DataFrame) -> pd.DataFrame:
    if routes.empty:
        return pd.DataFrame(columns=["port", "lat", "lon", "voyages", "tonnage"])
    origin = routes[["u", "u_lat", "u_lon", "voyage_count", "total_tonnage"]].rename(
        columns={"u": "port", "u_lat": "lat", "u_lon": "lon"}
    )
    dest = routes[["v", "v_lat", "v_lon", "voyage_count", "total_tonnage"]].rename(
        columns={"v": "port", "v_lat": "lat", "v_lon": "lon"}
    )
    ports = pd.concat([origin, dest], ignore_index=True)
    stats = (
        ports.groupby(["port", "lat", "lon"], as_index=False)
        .agg(voyages=("voyage_count", "sum"), tonnage=("total_tonnage", "sum"))
    )
    return stats


def _unique_highlight_ports(highlight_routes: pd.DataFrame, prefix: str) -> pd.DataFrame:
    col_map = {
        prefix: "port",
        f"{prefix}_lat": "lat",
        f"{prefix}_lon": "lon",
    }
    cols = [prefix, f"{prefix}_lat", f"{prefix}_lon", "highlight_score", "highlight_rank"]
    subset = highlight_routes[cols].rename(columns=col_map)
    unique = subset.drop_duplicates(subset=["port", "lat", "lon"]).reset_index(drop=True)
    return unique


def add_highlight_route_traces(
    fig: go.Figure,
    highlight_routes: pd.DataFrame,
    color: str,
    width: float,
    opacity: float,
) -> None:
    if highlight_routes.empty:
        return
    show_legend = True
    for row in highlight_routes.itertuples():
        source = getattr(row, "highlight_source", "current")
        hover = (
            f"Top {int(row.highlight_rank)} anomaly\n{row.u} → {row.v}<br>"
            f"Score: {row.highlight_score:.3f}<br>Geometry: {source}"
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[row.u_lon, row.v_lon],
                lat=[row.u_lat, row.v_lat],
                mode="lines",
                line=dict(width=width, color=color),
                opacity=opacity,
                hoverinfo="text",
                text=hover,
                name="Anomaly routes",
                legendgroup="anomalies",
                showlegend=show_legend,
            )
        )
        show_legend = False


def add_highlight_port_markers(fig: go.Figure, highlight_routes: pd.DataFrame, color: str, size: float) -> None:
    if highlight_routes.empty:
        return
    start_ports = _unique_highlight_ports(highlight_routes, "u")
    end_ports = _unique_highlight_ports(highlight_routes, "v")

    for ports, label, symbol in [
        (start_ports, "Anomaly origin", "star"),
        (end_ports, "Anomaly destination", "diamond"),
    ]:
        if ports.empty:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=ports["lon"],
                lat=ports["lat"],
                mode="markers+text",
                marker=dict(
                    symbol=symbol,
                    size=size,
                    color=color,
                    line=dict(color="#ffffff", width=1.2),
                ),
                text=ports["port"],
                textposition="top center",
                textfont=dict(color=color, size=10, family="Arial"),
                hoverinfo="text",
                name=label,
                showlegend=True,
            )
        )


def add_highlight_labels(fig: go.Figure, highlight_routes: pd.DataFrame, color: str) -> None:
    if highlight_routes.empty:
        return
    mid_lon = (highlight_routes["u_lon"].values + highlight_routes["v_lon"].values) / 2
    mid_lat = (highlight_routes["u_lat"].values + highlight_routes["v_lat"].values) / 2
    texts = [
        f"Top {int(rank)}: {u} → {v}<br>Score {score:.3f}"
        for u, v, rank, score in zip(
            highlight_routes["u"],
            highlight_routes["v"],
            highlight_routes["highlight_rank"],
            highlight_routes["highlight_score"],
        )
    ]
    fig.add_trace(
        go.Scattergeo(
            lon=mid_lon,
            lat=mid_lat,
            mode="text",
            text=texts,
            textfont=dict(color=color, size=9, family="Arial"),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def add_route_traces(fig: go.Figure, routes: pd.DataFrame, opacity: float) -> None:
    for row in routes.itertuples():
        hover = (
            f"{row.u} → {row.v}<br>Voyages: {row.voyage_count:,}<br>"
            f"Total tonnage: {row.total_tonnage:,.0f}<br>Score: {row.score:.2f}"
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[row.u_lon, row.v_lon],
                lat=[row.u_lat, row.v_lat],
                mode="lines",
                line=dict(width=row.line_width, color=row.line_color),
                opacity=opacity,
                hoverinfo="text",
                text=hover,
                showlegend=False,
            )
        )


def add_port_trace(fig: go.Figure, ports: pd.DataFrame, min_size: float, max_size: float) -> None:
    if ports.empty:
        return
    voyages_max = ports["voyages"].max() or 1.0
    tonnage_max = ports["tonnage"].max() or 1.0
    size_span = max_size - min_size
    marker_sizes = min_size + (ports["voyages"] / voyages_max) * size_span
    color_vals = ports["tonnage"] / tonnage_max
    fig.add_trace(
        go.Scattergeo(
            lon=ports["lon"],
            lat=ports["lat"],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=color_vals,
                colorscale="Sunset",
                showscale=True,
                colorbar=dict(title="Port tonnage<br>(relative)"),
                opacity=0.85,
            ),
            hoverinfo="text",
            text=[
                f"Port: {row.port}<br>Voyages: {row.voyages:,}<br>Total tonnage: {row.tonnage:,.0f}"
                for row in ports.itertuples()
            ],
            name="Ports",
        )
    )


def build_figure(snapshot_name: str, projection: str) -> go.Figure:
    fig = go.Figure()
    fig.update_geos(
        projection_type=projection,
        showcountries=True,
        showland=True,
        landcolor="#f4f1e1",
        oceancolor="#dce6f5",
        showocean=True,
        coastlinecolor="#888",
        showcoastlines=True,
    )
    fig.update_layout(
        title=dict(text=f"Shipping Routes — {snapshot_name}", x=0.5, xanchor="center"),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01,
            orientation="v",
            itemsizing="constant",
            tracegroupgap=5,
        ),
    )
    return fig


def render_snapshot(path: Path, args: argparse.Namespace, previous_snapshot_path: Path | None = None) -> Path:
    df = load_snapshot(path)
    routes = aggregate_routes(df)
    routes = score_routes(routes, args.count_weight, args.tonnage_weight)
    routes = filter_and_rank_routes(routes, args.min_score_quantile, args.max_routes)
    routes = assign_line_style(routes, args.min_line_width, args.max_line_width, args.line_colorscale)
    period_label = snapshot_period_from_stem(path.stem)
    highlights = load_top_anomalies(period_label, args)
    prev_df = load_snapshot(previous_snapshot_path) if previous_snapshot_path is not None else None
    highlight_routes = enrich_highlight_geometry(highlights, df, prev_df, period_label)
    ports = compute_port_stats(routes)

    fig = build_figure(path.stem, args.projection)
    add_route_traces(fig, routes, args.line_opacity)
    add_highlight_route_traces(fig, highlight_routes, args.highlight_line_color, args.highlight_line_width, opacity=0.9)
    add_highlight_port_markers(fig, highlight_routes, args.highlight_line_color, args.highlight_port_size)
    add_highlight_labels(fig, highlight_routes, args.highlight_line_color)
    add_port_trace(fig, ports, args.min_marker_size, args.max_marker_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{path.stem}_routes.html"
    fig.write_html(out_path)
    logging.info("Saved map to %s", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    snapshot_paths = list_snapshots(args.snapshot_dir)
    for idx, snapshot_path in enumerate(snapshot_paths):
        prev_path = snapshot_paths[idx - 1] if idx > 0 else None
        try:
            render_snapshot(snapshot_path, args, previous_snapshot_path=prev_path)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to render %s: %s", snapshot_path.name, exc)


if __name__ == "__main__":
    main()
