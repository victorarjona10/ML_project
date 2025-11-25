# Advanced Multi-Algorithm Music Recommendation System

## 📋 Overview

This is an **extremely sophisticated music recommendation system** that goes far beyond simple KNN recommendations. It uses **multiple machine learning algorithms in ensemble** to provide highly specific and accurate song recommendations.

## 🎯 Key Features

### 1. **Automatic Musical Profile Clustering**
- Instead of relying on genre labels (which aren't in the dataset), the system **automatically discovers 8 musical profiles** using KMeans clustering
- Each cluster is characterized by its unique combination of:
  - Danceability
  - Energy
  - Acousticness
  - Instrumentalness
  - Speechiness
  - Valence (mood)

### 2. **Cluster Types Identified**
The system automatically identifies these musical styles:

| Cluster Type | Characteristics | Key Features |
|--------------|-----------------|--------------|
| **Urban/Reggaeton/Hip-Hop** | High danceability, high speechiness | danceability, speechiness, energy, loudness, popularity |
| **Electronic/Dance/Pop** | High danceability, high energy | danceability, energy, valence, loudness, popularity |
| **Acoustic/Folk/Ballad** | High acousticness, low energy | acousticness, valence, instrumentalness, energy, popularity |
| **Classical/Jazz/Instrumental** | High instrumentalness | instrumentalness, acousticness, valence, duration_ms |
| **Rap/Hip-Hop/Spoken** | Very high speechiness | speechiness, danceability, energy, loudness, popularity |
| **Rock/Metal/Punk** | High energy, low acousticness | energy, loudness, valence, danceability |
| **Pop/R&B/Soul** | Medium danceability and energy | danceability, valence, energy, popularity, acousticness |
| **Melancholic/Sad/Alternative** | Low valence | valence, energy, acousticness, instrumentalness |

### 3. **Multi-Algorithm Ensemble**
For each recommendation, the system uses **4 different algorithms** and combines them:

#### Algorithm 1: K-Nearest Neighbors (KNN)
- Finds the most similar songs using Euclidean distance
- Optimized per cluster with appropriate neighbor counts
- Weight: 30%

#### Algorithm 2: Cosine Similarity
- Measures similarity based on feature vector angles
- Excellent for finding songs with similar "direction" in feature space
- Weight: 25%

#### Algorithm 3: Feature-Weighted Distance
- Uses cluster-specific feature importance
- For example:
  - Reggaeton songs prioritize: danceability, speechiness, energy
  - Acoustic songs prioritize: acousticness, valence, instrumentalness
  - Rock songs prioritize: energy, loudness, valence
- Weight: 30%

#### Algorithm 4: Popularity Adjustment
- Recommends songs with similar popularity levels
- Prevents always recommending only super-popular or unknown songs
- Weight: 15%

### 4. **Artist Diversity Enforcement**
- Limits maximum 2 songs per artist in recommendations
- Ensures variety in recommendations
- Prevents recommending an artist's entire discography

## 🔧 Technical Implementation

### Audio Features Analyzed

The system analyzes **11 key features** of each song:

1. **valence** (0-1): Musical positiveness/happiness
2. **acousticness** (0-1): How acoustic vs electronic
3. **danceability** (0-1): How suitable for dancing
4. **energy** (0-1): Intensity and activity
5. **instrumentalness** (0-1): Presence of vocals vs instruments
6. **liveness** (0-1): Presence of live audience
7. **loudness** (dB): Overall volume
8. **speechiness** (0-1): Presence of spoken words
9. **popularity** (0-100): Song's popularity score
10. **duration_ms**: Song length
11. **year**: Release year

### Architecture

```
Input Song
    ↓
Find Song in Dataset
    ↓
Identify Musical Cluster (1-8)
    ↓
Apply Cluster-Specific Feature Weights
    ↓
Run 4 Parallel Algorithms:
    - KNN in cluster
    - Cosine similarity
    - Weighted distance
    - Popularity matching
    ↓
Ensemble Results (weighted combination)
    ↓
Apply Artist Diversity Filter
    ↓
Return Top N Recommendations
```

## 📊 Why This Approach?

### Problem with Simple KNN
Simple KNN treats all features equally and doesn't understand musical context:
- A reggaeton song might be matched with electronic music just because both have high energy
- An acoustic ballad might match with jazz just because both are calm

### Our Solution
1. **Cluster First**: Group songs with similar overall musical profiles
2. **Specialize**: Use different feature weights per cluster type
3. **Ensemble**: Combine multiple algorithms to avoid biases of any single method
4. **Diversify**: Ensure variety in artist representation

## 🚀 Usage

### API Endpoint
```python
POST /recommend
{
    "song_name": "Despacito",
    "artist_name": "Luis Fonsi"  # optional but recommended
}
```

### Response
```json
{
    "song_found": {
        "name": "Despacito",
        "artist": "['Luis Fonsi', 'Daddy Yankee']",
        "year": 2017,
        "cluster": {
            "id": 3,
            "type": "Urban/Reggaeton/Hip-Hop",
            "key_features": ["danceability", "speechiness", "energy", "loudness", "popularity"]
        }
    },
    "recommendations": [
        {
            "name": "Similar Song 1",
            "artists": "...",
            "year": 2018,
            "popularity": 85,
            "similarity_score": 0.9234,
            "cluster_type": "Urban/Reggaeton/Hip-Hop"
        },
        ...
    ],
    "algorithm_info": {
        "method": "Multi-Algorithm Ensemble",
        "algorithms_used": [...],
        "cluster_based": true,
        "artist_diversity": true
    }
}
```

## 🎓 Machine Learning Algorithms Used

1. **Unsupervised Learning**:
   - KMeans Clustering (for musical profiles)
   - DBSCAN (available for alternative clustering)

2. **Similarity Metrics**:
   - Euclidean Distance
   - Cosine Similarity
   - Weighted Distance Metrics

3. **Ensemble Methods**:
   - Weighted voting of multiple algorithms
   - Cluster-specific model selection

4. **Feature Engineering**:
   - StandardScaler for normalization
   - Feature importance weighting per cluster

## 🎯 Optimization for Different Music Styles

### Reggaeton/Urban (High danceability + speechiness)
- **Primary**: Danceability, Speechiness, Energy
- **Secondary**: Loudness, Popularity
- **Why**: These songs are all about rhythm and vocal delivery

### Pop/Electronic (High danceability + energy)
- **Primary**: Danceability, Energy, Valence
- **Secondary**: Loudness, Popularity
- **Why**: Pop is about catchy, energetic, feel-good music

### Acoustic/Ballad (High acousticness)
- **Primary**: Acousticness, Valence, Instrumentalness
- **Secondary**: Energy (inverse), Popularity
- **Why**: Focus on emotional content and organic sound

### Rock/Metal (High energy + loudness)
- **Primary**: Energy, Loudness, Valence
- **Secondary**: Danceability
- **Why**: Intensity and power are key

## 📈 Performance Characteristics

- **Dataset Size**: 36,846 songs
- **Clustering**: 8 musical profiles
- **Features per Song**: 11 audio features
- **Algorithms per Recommendation**: 4 (KNN, Cosine, Weighted, Popularity)
- **Artist Diversity**: Max 2 songs per artist
- **Scalability**: O(n log n) with KNN indexing

## 🔮 Future Enhancements

Possible improvements:
1. Add temporal features (era-specific recommendations)
2. Include collaborative filtering (user behavior)
3. Deep learning embeddings (neural networks)
4. Real-time model updates
5. A/B testing framework for algorithm weights
6. User feedback loop for personalization

## 📝 Files

- `advanced_recommendation_engine.py` - Core recommendation engine
- `api_fase3.py` - FastAPI REST API
- `test_advanced_recommender.py` - Testing script

## 🎵 Example Use Cases

1. **Reggaeton fan**: Gets recommendations based on danceability and rhythm
2. **Classical listener**: Gets recommendations based on instrumentalness and complexity
3. **Mood-based**: Sad songs get sad recommendations (valence matching)
4. **Energy-based**: High energy songs get high energy recommendations
5. **Artist discovery**: Diversity filter ensures you discover new artists

---

**Built with**: Python, scikit-learn, FastAPI, pandas, numpy
**Version**: 2.0
**Date**: November 2025
