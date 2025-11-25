"""
Test script for the Advanced Recommendation Engine

This script demonstrates the capabilities of the new recommendation system
"""

from pathlib import Path
from advanced_recommendation_engine import AdvancedMusicRecommender
import pandas as pd

# Paths
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
METADATA_FILE = DATASETS_DIR / 'data_metadata.csv'
FEATURES_FILE = DATASETS_DIR / 'data_features_scaled.csv'

def main():
    print("=" * 80)
    print("TESTING ADVANCED MUSIC RECOMMENDATION ENGINE")
    print("=" * 80)
    
    # Initialize recommender
    print("\n1. Initializing recommender...")
    recommender = AdvancedMusicRecommender(
        metadata_path=str(METADATA_FILE),
        features_path=str(FEATURES_FILE),
        n_clusters=8
    )
    
    # Train the system
    print("\n2. Training the system...")
    recommender.train()
    
    # Test with different songs
    print("\n" + "=" * 80)
    print("3. TESTING RECOMMENDATIONS FOR DIFFERENT SONGS")
    print("=" * 80)
    
    test_songs = [
        ("Despacito", "Luis Fonsi"),  # Reggaeton/Latin
        ("Shape of You", "Ed Sheeran"),  # Pop
        ("Bohemian Rhapsody", "Queen"),  # Rock
        ("Old Town Road", "Lil Nas X"),  # Hip-Hop/Country fusion
    ]
    
    for song_name, artist_name in test_songs:
        print(f"\n{'=' * 80}")
        print(f"Testing: '{song_name}' by {artist_name}")
        print("=" * 80)
        
        # Find song
        idx = recommender.find_song_index(song_name, artist_name)
        
        if idx is None:
            print(f"[X] Song not found: '{song_name}' by {artist_name}")
            continue
        
        # Get song info
        song_info = recommender.df_metadata.iloc[idx]
        cluster_id = recommender.get_song_cluster(idx)
        cluster_info = recommender.cluster_profiles[cluster_id]
        
        print(f"\nFound Song:")
        print(f"   Name: {song_info['name']}")
        print(f"   Artist: {song_info['artists']}")
        print(f"   Year: {song_info['year']}")
        
        print(f"\nMusical Cluster:")
        print(f"   Cluster ID: {cluster_id}")
        print(f"   Type: {cluster_info['type']}")
        print(f"   Key Features: {', '.join(cluster_info['key_features'])}")
        print(f"   Cluster Size: {cluster_info['size']} songs")
        
        # Get recommendations
        print(f"\nTop 5 Recommendations:")
        recommendations = recommender.get_recommendations(idx, n_recommendations=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n   {i}. {rec['name']}")
            print(f"      Artist: {rec['artists']}")
            print(f"      Year: {rec['year']}")
            print(f"      Popularity: {rec['popularity']}/100")
            print(f"      Similarity Score: {rec['similarity_score']:.4f}")
            print(f"      Cluster Type: {rec['cluster_type']}")
    
    print("\n" + "=" * 80)
    print("[OK] Testing completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
