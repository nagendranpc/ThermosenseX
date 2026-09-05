"""
CLI Entry Point
Run the fire detection pipeline from the command line.
"""

import click
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import FireDetectionPipeline, load_config
from src.visualization.map_builder import build_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REGIONS = {
    "india":       "68,8,97,37",
    "middle_east": "35,12,65,38",
    "europe":      "-10,35,40,72",
    "siberia":     "60,50,120,75",
}


@click.command()
@click.option("--region",   default="india", type=click.Choice(list(REGIONS.keys()) + ["custom"]),
              help="Named region preset", show_default=True)
@click.option("--bbox",     default=None, help="Custom bbox: west,south,east,north (overrides --region)")
@click.option("--days",     default=7, type=int, help="Number of lookback days (1–10)", show_default=True)
@click.option("--sources",  default="VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT,MODIS_NRT",
              help="Comma-separated FIRMS sources")
@click.option("--esc-window",  default=14, type=int, help="Escalation rolling window (days)", show_default=True)
@click.option("--esc-zscore",  default=3.0, type=float, help="Escalation z-score threshold", show_default=True)
@click.option("--no-optical",  is_flag=True, default=False, help="Skip Sentinel-2 optical verification")
@click.option("--output-dir",  default="outputs", help="Output directory for map and report")
@click.option("--save-map",    is_flag=True, default=True, help="Save HTML map to output dir")
@click.option("--save-csv",    is_flag=True, default=True, help="Export classified CSV")
@click.option("--input-file", "--input-csv", "input_file", default=None,
              help="Path to local FIRMS data file (.zip, .shp, or .csv). Defaults to all files in data/raw/")
@click.option("--online", is_flag=True, default=False, help="Supplement local satellite data with NASA FIRMS live online API")
@click.option("--config", default="config/config.yaml", help="Config YAML path")
def main(region, bbox, days, sources, esc_window, esc_zscore, no_optical,
         output_dir, save_map, save_csv, input_file, online, config):
    """
    [FIRE] Industrial Fire Detection System - CLI

    \b
    Examples:
      python main.py --region india
      python main.py --region north_america --online
      python main.py --bbox "42,34,50,38"
    """
    # Resolve bbox
    resolved_bbox = bbox if bbox else REGIONS.get(region, REGIONS["india"])
    src_list      = [s.strip() for s in sources.split(",")]

    click.echo("\n" + "=" * 60)
    click.echo("  [FIRE] INDUSTRIAL FIRE DETECTION SYSTEM")
    click.echo("=" * 60)
    click.echo(f"  Region : {region} [{resolved_bbox}]")
    if input_file:
        click.echo(f"  Input  : {input_file}")
    else:
        click.echo("  Input  : Auto-loading all background satellite data (data/raw/)")
    if online:
        click.echo("  Online : Ingesting live NASA FIRMS API stream")
    click.echo(f"  Days   : {days}")
    click.echo(f"  Sources: {', '.join(src_list)}")
    click.echo(f"  Esc    : window={esc_window}d, z={esc_zscore}")
    click.echo("=" * 60 + "\n")

    # Load and patch config
    cfg = load_config(config)
    cfg.setdefault("escalation", {})
    cfg["escalation"]["window_days"]       = esc_window
    cfg["escalation"]["z_score_threshold"] = esc_zscore

    # Run pipeline
    pipeline = FireDetectionPipeline(cfg)
    results  = pipeline.run(
        bbox=resolved_bbox, days=days, sources=src_list,
        skip_optical=no_optical,
        local_csv=input_file,
        include_online_api=online,
    )

    fire_gdf = results["fire_gdf"]
    osm_gdf  = results["osm_gdf"]
    stats    = results["stats"]
    alerts   = results["alerts"]

    # ── Print summary ──────────────────────────────────────────────────────────
    click.echo("\n[SUMMARY] CLASSIFICATION RESULTS")
    click.echo("-" * 40)
    click.echo(f"  Total detections   : {stats['total_detections']}")
    click.echo(f"  Industrial Fires   : {stats['industrial_fires']}")
    click.echo(f"  Escalation Alerts  : {stats['escalation_alerts']}")
    click.echo(f"  Persistent Thermal : {stats['persistent_thermal']}")
    click.echo(f"  Wildfires          : {stats['wildfires']}")
    click.echo(f"  Agri Burns         : {stats['agricultural_burns']}")
    click.echo(f"  False Positives    : {stats['false_positives']}")
    click.echo(f"  Unknown            : {stats['unknown']}")
    click.echo(f"  Peak FRP           : {stats['max_frp_mw']} MW")

    # ── Escalation alerts ──────────────────────────────────────────────────────
    if not alerts.empty:
        click.echo(f"\n[ALERTS] ESCALATION EVENTS ({len(alerts)} site(s) - showing top 15)")
        click.echo("-" * 40)
        for _, row in alerts.head(15).iterrows():
            fac_name = str(row.get('osm_facility_name', '?')).encode('ascii', 'replace').decode('ascii')
            click.echo(
                f"  ! {fac_name} | "
                f"FRP {row.get('frp',0):.1f} MW (baseline {row.get('escalation_baseline_frp',0):.1f} MW) | "
                f"z={row.get('frp_zscore_vs_baseline',0):.2f} sigma"
            )

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datetime import date as ddate
    tag = f"{region}_{resolved_bbox.replace(',','_')}_{days}d_{ddate.today()}"

    if save_map:
        map_path = out_dir / "maps" / f"fire_map_{tag}.html"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        fmap = build_map(fire_gdf, osm_gdf, alerts)
        fmap.save(str(map_path))
        click.echo(f"\nMap saved to: {map_path}")

    if save_csv:
        class_col = "final_class" if "final_class" in fire_gdf.columns else "rule_class"
        csv_path  = out_dir / "reports" / f"detections_{tag}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        drop_cols = [c for c in fire_gdf.columns if c in ("geometry",)]
        fire_gdf.drop(columns=drop_cols, errors="ignore").to_csv(csv_path, index=False)
        click.echo(f"CSV saved to: {csv_path}")

    click.echo("\nDone.\n")


if __name__ == "__main__":
    main()
