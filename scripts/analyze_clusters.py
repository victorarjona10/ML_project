"""
Cluster Analysis and Visualization
Helps understand the musical profiles discovered by the system
"""

import sys
sys.path.append('c:/Users/mohal/UNI/4A/RAIA/ML_project/scripts')

from pathlib import Path
from advanced_recommendation_engine import get_recommender_instance
import pandas as pd

# Initialize the recommender
DATASETS_DIR = Path('c:/Users/mohal/UNI/4A/RAIA/ML_project/datasets')
METADATA_FILE = DATASETS_DIR / 'data_metadata.csv'
FEATURES_FILE = DATASETS_DIR / 'data_features_scaled.csv'

print("="*80)
print("MUSICAL CLUSTER ANALYSIS")
print("="*80)

# Get recommender instance
recommender = get_recommender_instance(
    metadata_path=str(METADATA_FILE),
    features_path=str(FEATURES_FILE),
    n_clusters=8
)

print("\n" + "="*80)
print("DETAILED CLUSTER PROFILES")
print("="*80)

for cluster_id in range(8):
    profile = recommender.cluster_profiles[cluster_id]
    
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id}: {profile['type']}")
    print("="*80)
    
    print(f"\nSize: {profile['size']} songs ({profile['size']/36846*100:.1f}% of dataset)")
    
    print(f"\nAverage Characteristics:")
    print(f"  Danceability:     {profile['avg_danceability']:.3f}")
    print(f"  Energy:           {profile['avg_energy']:.3f}")
    print(f"  Acousticness:     {profile['avg_acousticness']:.3f}")
    print(f"  Instrumentalness: {profile['avg_instrumentalness']:.3f}")
    print(f"  Speechiness:      {profile['avg_speechiness']:.3f}")
    print(f"  Valence (mood):   {profile['avg_valence']:.3f}")
    print(f"  Popularity:       {profile['avg_popularity']:.1f}/100")
    
    print(f"\nKey Features for Recommendations:")
    for i, feature in enumerate(profile['key_features'], 1):
        print(f"  {i}. {feature}")
    
    # Get some example songs from this cluster
    cluster_mask = recommender.cluster_labels == cluster_id
    cluster_indices = recommender.df_metadata[cluster_mask].head(10).index
    
    print(f"\nExample Songs in this Cluster:")
    for i, idx in enumerate(cluster_indices, 1):
        song = recommender.df_metadata.iloc[idx]
        print(f"  {i}. '{song['name']}' by {song['artists']} ({song['year']})")

print("\n" + "="*80)
print("CLUSTER COMPARISON SUMMARY")
print("="*80)

# Create summary table
summary_data = []
for cluster_id in range(8):
    profile = recommender.cluster_profiles[cluster_id]
    summary_data.append({
        'Cluster': f"{cluster_id}: {profile['type'][:30]}",
        'Size': profile['size'],
        '%': f"{profile['size']/36846*100:.1f}%",
        'Dance': f"{profile['avg_danceability']:.2f}",
        'Energy': f"{profile['avg_energy']:.2f}",
        'Acoustic': f"{profile['avg_acousticness']:.2f}",
        'Speech': f"{profile['avg_speechiness']:.2f}",
        'Mood': f"{profile['avg_valence']:.2f}",
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. The system automatically discovered 8 distinct musical profiles
2. Each cluster has unique audio characteristics that define its style
3. Recommendations within each cluster use specialized feature weights
4. This ensures reggaeton gets reggaeton recommendations, not just "high energy" music

Feature Guide:
  - Danceability: 0 (not danceable) to 1 (very danceable)
  - Energy: 0 (calm) to 1 (intense)
  - Acousticness: 0 (electronic) to 1 (acoustic)
  - Speechiness: 0 (instrumental) to 1 (spoken word/rap)
  - Mood (valence): 0 (sad) to 1 (happy)
""")

print("="*80)
print("Analysis complete!")
print("="*80)
