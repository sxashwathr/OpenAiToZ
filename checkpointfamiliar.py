import rasterio
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = ""

# set openAI Key
client = OpenAI(api_key=os.getenv(""))

# load LIDAR dataset
dataset_path = os.path.expanduser("~/Downloads/output_be.tif")  
with rasterio.open(dataset_path) as src:
    elevation_data = src.read(1) 
    profile = src.profile
    dataset_id = src.name

# clean datapoints
valid_mask = (elevation_data > -9999) & (elevation_data < 9999) 
cleaned_data = elevation_data.copy()
cleaned_data[~valid_mask] = np.nan 

# elevation data
plt.figure(figsize=(10, 8))
plt.imshow(cleaned_data, cmap='terrain')
plt.title('Elevation Preview')
plt.colorbar(label='Elevation (m)')
plt.show()

# summarize data to send to model
if valid_mask.any():
    valid_data = elevation_data[valid_mask]

    if len(valid_data) > 0:
        summary_stats = {
            "min_elevation": float(np.min(valid_data)),
            "max_elevation": float(np.max(valid_data)),
            "mean_elevation": float(np.mean(valid_data)),
            "median_elevation": float(np.median(valid_data)),
            "std_elevation": float(np.std(valid_data)),
            "elevation_range": float(np.max(valid_data) - np.min(valid_data)),
            "data_coverage": float(len(valid_data)/elevation_data.size)*100
        }

    terrain_summary = f"""
Elevation Data Analysis:
- Elevation range: {summary_stats['min_elevation']:.1f}m to {summary_stats['max_elevation']:.1f}m
- Average elevation: {summary_stats['mean_elevation']:.1f}m
- Median elevation: {summary_stats['median_elevation']:.1f}m
- Terrain variation (std dev): {summary_stats['std_elevation']:.1f}m
- Data coverage: {summary_stats['data_coverage']:.1f}% of area

This appears to be {"mountainous" if summary_stats['std_elevation'] > 200 else "relatively flat"} terrain
with {"high" if summary_stats['mean_elevation'] > 1000 else "moderate" if summary_stats['mean_elevation'] > 200 else "low"} average elevation.
"""

    print("=== TERRAIN ANALYSIS COMPLETE ===")
    print(terrain_summary)

    
    print("=== READY FOR AI ANALYSIS ===")
    print("Summary prepared for OpenAI model:")
    print(terrain_summary)

    prompt = f"Given this elevation data summary: {terrain_summary}, describe the surface features in plain English."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful geospatial analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    model_used = response.model
    model_output = response.choices[0].message.content.strip()

    # Checkpoint results
    print("Model used:", model_used)
    print("Dataset:", dataset_id)
    print("Model response:\n", model_output)


else:
    print("ERROR: No valid elevation data found in this file.")
    terrain_summary = "No valid elevation data available for analysis."
