"""
Advanced Multi-Algorithm Music Recommendation Engine
=====================================================

This module implements a sophisticated recommendation system that:
1. Clusters songs into musical profiles using KMeans
2. Applies specialized algorithms per cluster
3. Uses ensemble methods for precise recommendations
4. Considers multiple audio features: acousticness, danceability, energy, 
   instrumentalness, popularity, valence, speechiness, etc.

Author: Advanced ML Recommendation System
Date: 2025-11-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class AdvancedMusicRecommender:
    """
    Advanced recommendation system using multiple algorithms and clustering
    """
    
    def __init__(self, metadata_path, features_path, n_clusters=8):
        """
        Initialize the recommender system
        
        Args:
            metadata_path: Path to metadata CSV
            features_path: Path to features CSV (scaled)
            n_clusters: Number of musical profile clusters to create
        """
        self.n_clusters = n_clusters
        self.metadata_path = metadata_path
        self.features_path = features_path
        
        # Data storage
        self.df_metadata = None
        self.df_features = None
        self.df_features_scaled = None
        self.feature_columns = None
        
        # Models
        self.kmeans_model = None
        self.cluster_labels = None
        self.cluster_models = {}  # Different models per cluster
        
        # Cluster profiles (to understand what each cluster represents)
        self.cluster_profiles = {}
        
    def load_data(self):
        """Load and prepare the datasets"""
        print("Loading datasets...")
        
        self.df_metadata = pd.read_csv(self.metadata_path)
        features_df = pd.read_csv(self.features_path)
        
        # Clean metadata
        self.df_metadata['name'] = self.df_metadata['name'].astype(str)
        self.df_metadata['artists'] = self.df_metadata['artists'].astype(str)
        
        # For features, we need the original non-scaled data for some algorithms
        # Load the filtered data to get original features
        filtered_path = Path(self.metadata_path).parent / 'data_filtered.csv'
        df_original = pd.read_csv(filtered_path)
        
        # Extract relevant feature columns
        self.feature_columns = [
            'valence', 'acousticness', 'danceability', 'energy',
            'instrumentalness', 'liveness', 'loudness', 'speechiness',
            'popularity', 'duration_ms', 'year'
        ]
        
        # Create feature matrix from original data
        self.df_features = df_original[self.feature_columns].copy()
        
        # Scale features for algorithms that need it
        scaler = StandardScaler()
        self.df_features_scaled = pd.DataFrame(
            scaler.fit_transform(self.df_features),
            columns=self.feature_columns,
            index=self.df_features.index
        )
        
        print(f"[OK] Loaded {len(self.df_metadata)} songs with {len(self.feature_columns)} features")
        
    def create_musical_clusters(self):
        """
        Create clusters of songs based on their musical characteristics
        This replaces genre-based classification
        """
        print(f"Creating {self.n_clusters} musical profile clusters...")
        
        # Use specific features for clustering to create meaningful groups
        clustering_features = [
            'acousticness', 'danceability', 'energy', 
            'instrumentalness', 'speechiness', 'valence'
        ]
        
        X_cluster = self.df_features_scaled[clustering_features].values
        
        # KMeans clustering
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=20,
            max_iter=500
        )
        self.cluster_labels = self.kmeans_model.fit_predict(X_cluster)
        
        # Analyze cluster profiles
        self._analyze_clusters()
        
        print("[OK] Musical clusters created successfully")
        
    def _analyze_clusters(self):
        """Analyze and profile each cluster to understand its characteristics"""
        print("\n[*] Analyzing cluster profiles...")
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_data = self.df_features[cluster_mask]
            
            # Calculate mean characteristics
            profile = {
                'size': cluster_mask.sum(),
                'avg_danceability': cluster_data['danceability'].mean(),
                'avg_energy': cluster_data['energy'].mean(),
                'avg_acousticness': cluster_data['acousticness'].mean(),
                'avg_instrumentalness': cluster_data['instrumentalness'].mean(),
                'avg_speechiness': cluster_data['speechiness'].mean(),
                'avg_valence': cluster_data['valence'].mean(),
                'avg_popularity': cluster_data['popularity'].mean(),
            }
            
            # Determine cluster type based on characteristics
            cluster_type = self._determine_cluster_type(profile)
            profile['type'] = cluster_type
            profile['key_features'] = self._get_key_features_for_cluster(profile)
            
            self.cluster_profiles[cluster_id] = profile
            
            print(f"\n  Cluster {cluster_id} ({cluster_type}):")
            print(f"    - Size: {profile['size']} songs")
            print(f"    - Danceability: {profile['avg_danceability']:.3f}")
            print(f"    - Energy: {profile['avg_energy']:.3f}")
            print(f"    - Acousticness: {profile['avg_acousticness']:.3f}")
            print(f"    - Valence: {profile['avg_valence']:.3f}")
            print(f"    - Key features: {', '.join(profile['key_features'])}")
    
    def _determine_cluster_type(self, profile):
        """Determine the musical style/type based on cluster characteristics"""
        dance = profile['avg_danceability']
        energy = profile['avg_energy']
        acoustic = profile['avg_acousticness']
        instrumental = profile['avg_instrumentalness']
        speech = profile['avg_speechiness']
        valence = profile['avg_valence']
        
        # High danceability + high energy = Electronic/Pop/Reggaeton style
        if dance > 0.65 and energy > 0.7:
            if speech > 0.15:
                return "Urban/Reggaeton/Hip-Hop"
            else:
                return "Electronic/Dance/Pop"
        
        # High acousticness + low energy = Acoustic/Folk/Ballad
        elif acoustic > 0.5 and energy < 0.5:
            return "Acoustic/Folk/Ballad"
        
        # High instrumentalness = Classical/Jazz/Instrumental
        elif instrumental > 0.3:
            return "Classical/Jazz/Instrumental"
        
        # High speechiness = Rap/Spoken Word
        elif speech > 0.3:
            return "Rap/Hip-Hop/Spoken"
        
        # High energy + low acousticness = Rock/Metal
        elif energy > 0.7 and acoustic < 0.2:
            return "Rock/Metal/Punk"
        
        # Medium danceability + medium energy = Pop/R&B
        elif 0.5 <= dance <= 0.7 and 0.5 <= energy <= 0.7:
            return "Pop/R&B/Soul"
        
        # Low valence = Sad/Melancholic
        elif valence < 0.3:
            return "Melancholic/Sad/Alternative"
        
        else:
            return "Mixed/Alternative"
    
    def _get_key_features_for_cluster(self, profile):
        """
        Determine which features are most important for this cluster type
        This will be used to weight features in recommendations
        """
        cluster_type = profile['type']
        
        # Define important features per cluster type
        feature_importance = {
            "Urban/Reggaeton/Hip-Hop": [
                'danceability', 'speechiness', 'energy', 'loudness', 'popularity'
            ],
            "Electronic/Dance/Pop": [
                'danceability', 'energy', 'valence', 'loudness', 'popularity'
            ],
            "Acoustic/Folk/Ballad": [
                'acousticness', 'valence', 'instrumentalness', 'energy', 'popularity'
            ],
            "Classical/Jazz/Instrumental": [
                'instrumentalness', 'acousticness', 'valence', 'duration_ms'
            ],
            "Rap/Hip-Hop/Spoken": [
                'speechiness', 'danceability', 'energy', 'loudness', 'popularity'
            ],
            "Rock/Metal/Punk": [
                'energy', 'loudness', 'valence', 'danceability'
            ],
            "Pop/R&B/Soul": [
                'danceability', 'valence', 'energy', 'popularity', 'acousticness'
            ],
            "Melancholic/Sad/Alternative": [
                'valence', 'energy', 'acousticness', 'instrumentalness'
            ],
            "Mixed/Alternative": [
                'energy', 'valence', 'danceability', 'popularity'
            ]
        }
        
        return feature_importance.get(cluster_type, ['energy', 'valence', 'danceability'])
    
    def train_cluster_specific_models(self):
        """
        Train specialized models for each cluster
        Different musical styles need different recommendation strategies
        """
        print("\nTraining specialized models for each cluster...")
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_type = self.cluster_profiles[cluster_id]['type']
            key_features = self.cluster_profiles[cluster_id]['key_features']
            
            print(f"\n  Training models for Cluster {cluster_id} ({cluster_type})...")
            
            # Get cluster data
            X_cluster = self.df_features_scaled[cluster_mask][key_features].values
            
            # Train multiple models for this cluster
            models = {}
            
            # 1. KNN - good for finding similar songs
            knn = NearestNeighbors(
                n_neighbors=min(50, cluster_mask.sum()),
                metric='euclidean',
                algorithm='auto'
            )
            knn.fit(X_cluster)
            models['knn'] = knn
            
            # 2. Cosine similarity matrix - good for feature-based similarity
            models['cosine_matrix'] = cosine_similarity(X_cluster)
            
            # 3. Store cluster indices and features for later use
            models['indices'] = np.where(cluster_mask)[0]
            models['features'] = X_cluster
            models['key_features'] = key_features
            
            self.cluster_models[cluster_id] = models
            
            print(f"    [OK] Trained models with {len(models['indices'])} songs")
    
    def get_song_cluster(self, song_index):
        """Get the cluster ID for a given song"""
        return self.cluster_labels[song_index]
    
    def recommend_multi_algorithm(self, song_index, n_recommendations=10):
        """
        Generate recommendations using multiple algorithms and ensemble them
        
        This is the main recommendation function that combines:
        - KNN in the same cluster
        - Cosine similarity
        - Feature-weighted distance
        - Popularity boosting
        - Artist diversity
        
        Args:
            song_index: Index of the song to recommend from
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended song indices with scores
        """
        # Get song's cluster
        cluster_id = self.get_song_cluster(song_index)
        cluster_type = self.cluster_profiles[cluster_id]['type']
        
        print(f"\n[*] Recommending for song in cluster {cluster_id} ({cluster_type})")
        
        # Get cluster models
        cluster_models = self.cluster_models[cluster_id]
        key_features = cluster_models['key_features']
        cluster_indices = cluster_models['indices']
        
        # Find position of song in cluster
        song_cluster_position = np.where(cluster_indices == song_index)[0]
        
        if len(song_cluster_position) == 0:
            # Song not in its expected cluster (shouldn't happen)
            print("[!] Warning: Song not in expected cluster, using global search")
            return self._fallback_recommendation(song_index, n_recommendations)
        
        song_cluster_position = song_cluster_position[0]
        
        # Get song features
        song_features = self.df_features_scaled.iloc[song_index][key_features].values.reshape(1, -1)
        
        # Algorithm 1: KNN-based recommendations
        knn_scores = self._get_knn_recommendations(
            song_cluster_position, cluster_models, n_recommendations * 3
        )
        
        # Algorithm 2: Cosine similarity
        cosine_scores = self._get_cosine_recommendations(
            song_cluster_position, cluster_models, n_recommendations * 3
        )
        
        # Algorithm 3: Feature-weighted Euclidean distance
        weighted_scores = self._get_weighted_recommendations(
            song_index, cluster_id, cluster_indices, n_recommendations * 3
        )
        
        # Algorithm 4: Popularity-adjusted scores
        popularity_scores = self._get_popularity_adjusted_scores(
            cluster_indices, song_index
        )
        
        # Ensemble all scores
        final_scores = self._ensemble_scores(
            knn_scores, cosine_scores, weighted_scores, 
            popularity_scores, cluster_indices
        )
        
        # Apply artist diversity
        final_recommendations = self._apply_artist_diversity(
            final_scores, song_index, n_recommendations
        )
        
        return final_recommendations
    
    def _get_knn_recommendations(self, song_cluster_position, cluster_models, k):
        """Get KNN-based recommendations"""
        knn_model = cluster_models['knn']
        cluster_features = cluster_models['features']
        
        song_features = cluster_features[song_cluster_position].reshape(1, -1)
        distances, indices = knn_model.kneighbors(song_features, n_neighbors=k+1)
        
        # Skip first one (the song itself)
        scores = {}
        for idx, dist in zip(indices[0][1:], distances[0][1:]):
            # Convert distance to similarity (inverse)
            similarity = 1 / (1 + dist)
            scores[idx] = similarity
            
        return scores
    
    def _get_cosine_recommendations(self, song_cluster_position, cluster_models, k):
        """Get cosine similarity-based recommendations"""
        cosine_matrix = cluster_models['cosine_matrix']
        
        # Get similarities for this song
        similarities = cosine_matrix[song_cluster_position]
        
        # Get top k (excluding the song itself)
        top_indices = np.argsort(similarities)[::-1][1:k+1]
        
        scores = {}
        for idx in top_indices:
            scores[idx] = similarities[idx]
            
        return scores
    
    def _get_weighted_recommendations(self, song_index, cluster_id, cluster_indices, k):
        """
        Get recommendations using feature-weighted distance
        Different features have different importance per cluster
        """
        key_features = self.cluster_profiles[cluster_id]['key_features']
        
        # Get song features
        song_features = self.df_features_scaled.iloc[song_index][key_features].values
        
        # Calculate weighted distances to all songs in cluster
        scores = {}
        for cluster_pos, global_idx in enumerate(cluster_indices):
            if global_idx == song_index:
                continue
                
            candidate_features = self.df_features_scaled.iloc[global_idx][key_features].values
            
            # Weighted Euclidean distance
            distance = np.sqrt(np.sum((song_features - candidate_features) ** 2))
            similarity = 1 / (1 + distance)
            
            scores[cluster_pos] = similarity
        
        # Get top k
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return dict(top_items)
    
    def _get_popularity_adjusted_scores(self, cluster_indices, song_index):
        """
        Adjust scores based on popularity
        More popular songs get a small boost (but not too much to avoid only recommending hits)
        """
        song_popularity = self.df_features.iloc[song_index]['popularity']
        
        scores = {}
        for cluster_pos, global_idx in enumerate(cluster_indices):
            if global_idx == song_index:
                continue
                
            candidate_popularity = self.df_features.iloc[global_idx]['popularity']
            
            # Boost similar popularity songs slightly
            popularity_diff = abs(song_popularity - candidate_popularity)
            popularity_score = 1 / (1 + popularity_diff / 100)  # Normalize
            
            scores[cluster_pos] = popularity_score
            
        return scores
    
    def _ensemble_scores(self, knn_scores, cosine_scores, weighted_scores, 
                        popularity_scores, cluster_indices):
        """
        Combine all scores using weighted ensemble
        
        Weights can be adjusted based on empirical performance
        """
        # Weights for each algorithm
        weights = {
            'knn': 0.30,
            'cosine': 0.25,
            'weighted': 0.30,
            'popularity': 0.15
        }
        
        # Combine scores
        final_scores = {}
        
        # Get all unique indices from all algorithms
        all_indices = set()
        all_indices.update(knn_scores.keys())
        all_indices.update(cosine_scores.keys())
        all_indices.update(weighted_scores.keys())
        all_indices.update(popularity_scores.keys())
        
        for cluster_pos in all_indices:
            score = 0
            score += knn_scores.get(cluster_pos, 0) * weights['knn']
            score += cosine_scores.get(cluster_pos, 0) * weights['cosine']
            score += weighted_scores.get(cluster_pos, 0) * weights['weighted']
            score += popularity_scores.get(cluster_pos, 0) * weights['popularity']
            
            # Map back to global index
            global_idx = cluster_indices[cluster_pos]
            final_scores[global_idx] = score
        
        return final_scores
    
    def _apply_artist_diversity(self, scores, song_index, n_recommendations):
        """
        Apply artist diversity to avoid recommending too many songs from the same artist
        """
        song_artist = self.df_metadata.iloc[song_index]['artists']
        
        # Sort by score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        artist_count = {}
        
        for global_idx, score in sorted_candidates:
            if len(recommendations) >= n_recommendations:
                break
                
            candidate_artist = self.df_metadata.iloc[global_idx]['artists']
            
            # Limit songs per artist (max 2)
            if artist_count.get(candidate_artist, 0) < 2:
                recommendations.append({
                    'index': global_idx,
                    'score': score,
                    'artist': candidate_artist
                })
                artist_count[candidate_artist] = artist_count.get(candidate_artist, 0) + 1
        
        return recommendations
    
    def _fallback_recommendation(self, song_index, n_recommendations):
        """Fallback method if cluster-based recommendation fails"""
        # Use global KNN
        song_features = self.df_features_scaled.iloc[song_index].values.reshape(1, -1)
        
        knn_global = NearestNeighbors(n_neighbors=n_recommendations+1, metric='euclidean')
        knn_global.fit(self.df_features_scaled.values)
        
        distances, indices = knn_global.kneighbors(song_features)
        
        recommendations = []
        for idx, dist in zip(indices[0][1:], distances[0][1:]):
            recommendations.append({
                'index': idx,
                'score': 1 / (1 + dist),
                'artist': self.df_metadata.iloc[idx]['artists']
            })
        
        return recommendations
    
    def find_song_index(self, song_name, artist_name=""):
        """Find song index by name and optionally artist"""
        if self.df_metadata is None:
            return None
        
        # Search by song name
        song_matches = self.df_metadata[
            self.df_metadata['name'].str.contains(song_name, case=False, na=False)
        ]
        
        if song_matches.empty:
            return None
        
        # Filter by artist if provided
        if artist_name:
            artist_matches = song_matches[
                song_matches['artists'].str.contains(artist_name, case=False, na=False)
            ]
            
            if not artist_matches.empty:
                return artist_matches.index[0]
            else:
                return None
        
        return song_matches.index[0]
    
    def get_recommendations(self, song_index, n_recommendations=5):
        """
        Main public method to get recommendations
        
        Args:
            song_index: Index of the song
            n_recommendations: Number of recommendations
            
        Returns:
            List of dictionaries with recommendation details
        """
        recommendations = self.recommend_multi_algorithm(song_index, n_recommendations)
        
        results = []
        for rec in recommendations:
            song_data = self.df_metadata.iloc[rec['index']]
            similarity_score = float(rec['score'])
            
            results.append({
                'name': song_data['name'],
                'artists': song_data['artists'],
                'year': int(song_data['year']),
                'popularity': int(self.df_features.iloc[rec['index']]['popularity']),
                'similarity_score': similarity_score,
                'similarity_percentage': round(similarity_score * 100, 2),  # Convert to percentage
                'cluster_type': self.cluster_profiles[self.get_song_cluster(rec['index'])]['type']
            })
        
        return results
    
    def train(self):
        """Train the complete recommendation system"""
        self.load_data()
        self.create_musical_clusters()
        self.train_cluster_specific_models()
        print("\n[OK] Advanced recommendation system ready!")


# Singleton instance for API usage
_recommender_instance = None

def get_recommender_instance(metadata_path=None, features_path=None, n_clusters=8):
    """Get or create the recommender instance"""
    global _recommender_instance
    
    if _recommender_instance is None:
        if metadata_path is None or features_path is None:
            raise ValueError("Must provide paths for first initialization")
        
        _recommender_instance = AdvancedMusicRecommender(
            metadata_path, features_path, n_clusters
        )
        _recommender_instance.train()
    
    return _recommender_instance
