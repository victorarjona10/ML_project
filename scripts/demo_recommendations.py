"""
Example demonstration of the Advanced Music Recommendation System
Shows how different musical styles get different recommendations
"""

import sys
sys.path.append('c:/Users/mohal/UNI/4A/RAIA/ML_project/scripts')

from pathlib import Path
from advanced_recommendation_engine import get_recommender_instance

# Initialize the recommender
DATASETS_DIR = Path('c:/Users/mohal/UNI/4A/RAIA/ML_project/datasets')
METADATA_FILE = DATASETS_DIR / 'data_metadata.csv'
FEATURES_FILE = DATASETS_DIR / 'data_features_scaled.csv'

print("=" * 80)
print("ADVANCED MUSIC RECOMMENDATION SYSTEM - DEMONSTRATION")
print("=" * 80)

# Get recommender instance (will train on first call)
recommender = get_recommender_instance(
    metadata_path=str(METADATA_FILE),
    features_path=str(FEATURES_FILE),
    n_clusters=8
)

def show_recommendations(song_name, artist_name=""):
    """Helper function to display recommendations"""
    print(f"\n{'='*80}")
    print(f"Song: '{song_name}'" + (f" by {artist_name}" if artist_name else ""))
    print("="*80)
    
    idx = recommender.find_song_index(song_name, artist_name)
    
    if idx is None:
        print(f"[!] Song not found in database")
        return
    
    # Get song info
    song = recommender.df_metadata.iloc[idx]
    cluster_id = recommender.get_song_cluster(idx)
    cluster_info = recommender.cluster_profiles[cluster_id]
    
    # Display song details
    print(f"\nFound: {song['name']} by {song['artists']} ({song['year']})")
    print(f"Cluster: {cluster_info['type']}")
    print(f"Key Features for this cluster: {', '.join(cluster_info['key_features'])}")
    
    # Get and display recommendations
    recs = recommender.get_recommendations(idx, n_recommendations=5)
    
    print(f"\nTop 5 Recommendations (using multi-algorithm ensemble):")
    for i, rec in enumerate(recs, 1):
        print(f"\n  {i}. {rec['name']}")
        print(f"     Artist: {rec['artists']}")
        print(f"     Year: {rec['year']} | Popularity: {rec['popularity']}/100")
        print(f"     Similarity Score: {rec['similarity_score']:.4f}")
        print(f"     Type: {rec['cluster_type']}")

# Test different musical styles
print("\n\n" + "="*80)
print("TESTING DIFFERENT MUSICAL STYLES")
print("="*80)

# You can try these - uncomment any that exist in your dataset
test_cases = [
    # Pop/Electronic
    ("Shape of You", "Ed Sheeran"),
    
    # Rock
    ("All The Small Things", "blink-182"),
    
    # Hip-Hop/Rap
    ("Still D.R.E.", "Dr. Dre"),
    
    # R&B/Soul
    ("No Scrubs", "TLC"),
]

for song, artist in test_cases:
    try:
        show_recommendations(song, artist)
    except Exception as e:
        print(f"\n[ERROR] {e}")

print("\n" + "="*80)
print("DEMONSTRATION COMPLETE")
print("="*80)
print("\nKey Insights:")
print("- Each song is assigned to a musical cluster based on its audio features")
print("- Different clusters use different feature weights for recommendations")
print("- Multiple algorithms (KNN, Cosine Similarity, Weighted Distance, Popularity)")
print("  are combined to provide the best recommendations")
print("- Artist diversity is enforced to avoid recommending too many songs from one artist")
