import sys, os
sys.path.insert(0, '.')
os.environ['USE_MOCK_DATA'] = 'true'

from src.pipeline import FireDetectionPipeline, load_config

cfg      = load_config('config/config.yaml')
pipeline = FireDetectionPipeline(cfg)
results  = pipeline.run(bbox='68,8,97,37', days=3, skip_optical=True)

gdf   = results['fire_gdf']
stats = results['stats']

print("=== PIPELINE RESULTS ===")
print(f"Total detections  : {stats['total_detections']}")
print(f"Industrial Fires  : {stats['industrial_fires']}")
print(f"Escalations       : {stats['escalation_alerts']}")
print(f"Persistent Thermal: {stats['persistent_thermal']}")
print(f"Wildfires         : {stats['wildfires']}")
print(f"Agricultural Burns: {stats['agricultural_burns']}")
print(f"False Positives   : {stats['false_positives']}")
print(f"Peak FRP          : {stats['max_frp_mw']} MW")
print()
print("=== CLASS DISTRIBUTION ===")
print(gdf['final_class'].value_counts())
print()

# Test map generation
from src.visualization.map_builder import build_map
from src.ingestion.osm_client import OSMClient
osm = OSMClient(cfg)
osm_gdf = osm.fetch_facilities('68,8,97,37')
fmap = build_map(gdf, osm_gdf)
fmap.save('outputs/maps/test_map.html')
print("Map saved to outputs/maps/test_map.html")
print("ALL TESTS PASSED")
