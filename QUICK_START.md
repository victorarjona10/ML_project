# Quick Start Guide - Advanced Music Recommendation API

## Installation

1. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Running the API

1. Navigate to the scripts folder:
```powershell
cd c:\Users\mohal\UNI\4A\RAIA\ML_project\scripts
```

2. Start the API server:
```powershell
python api_fase3.py
```

3. The API will be available at: `http://127.0.0.1:8000`

4. Open the interactive documentation at: `http://127.0.0.1:8000/docs`

## Testing the API

### Using the browser (Swagger UI)
1. Go to `http://127.0.0.1:8000/docs`
2. Click on `POST /recommend`
3. Click "Try it out"
4. Enter your song request:
```json
{
  "song_name": "Despacito",
  "artist_name": "Luis Fonsi"
}
```
5. Click "Execute"

### Using curl
```powershell
curl -X POST "http://127.0.0.1:8000/recommend" `
  -H "Content-Type: application/json" `
  -d '{"song_name": "Shape of You", "artist_name": "Ed Sheeran"}'
```

### Using Python requests
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/recommend",
    json={
        "song_name": "Bohemian Rhapsody",
        "artist_name": "Queen"
    }
)

print(response.json())
```

## Example Songs to Try

Here are some songs from the dataset you can test:

- **Pop**: "Shape of You" by "Ed Sheeran"
- **Rock**: "All The Small Things" by "blink-182"
- **Hip-Hop**: "Still D.R.E." by "Dr. Dre"
- **Christmas**: "It's Beginning to Look a Lot Like Christmas" by "Perry Como"
- **R&B**: "No Scrubs" by "TLC"

## API Response Format

```json
{
  "song_found": {
    "name": "Shape of You",
    "artist": "['Ed Sheeran']",
    "year": 2017,
    "cluster": {
      "id": 1,
      "type": "Electronic/Dance/Pop",
      "key_features": ["danceability", "energy", "valence", "loudness", "popularity"]
    }
  },
  "recommendations": [
    {
      "name": "Similar Song 1",
      "artists": "Artist Name",
      "year": 2018,
      "popularity": 85,
      "similarity_score": 0.9234,
      "cluster_type": "Electronic/Dance/Pop"
    }
  ],
  "algorithm_info": {
    "method": "Multi-Algorithm Ensemble",
    "algorithms_used": [
      "K-Nearest Neighbors (KNN)",
      "Cosine Similarity",
      "Feature-Weighted Distance",
      "Popularity Adjustment"
    ],
    "cluster_based": true,
    "artist_diversity": true
  }
}
```

## System Features

### Musical Clusters Identified
The system automatically groups songs into these types:
- **Rock/Metal/Punk**: High energy, loud
- **Electronic/Dance/Pop**: High danceability and energy
- **Acoustic/Folk/Ballad**: High acousticness, calm
- **Rap/Hip-Hop/Spoken**: High speechiness
- **Pop/R&B/Soul**: Balanced characteristics
- **Melancholic/Sad**: Low valence (mood)
- **Classical/Jazz/Instrumental**: High instrumentalness

### Algorithms Used
Each recommendation uses 4 algorithms in ensemble:
1. **K-Nearest Neighbors (KNN)** - Euclidean distance similarity
2. **Cosine Similarity** - Angular similarity between feature vectors
3. **Feature-Weighted Distance** - Cluster-specific feature importance
4. **Popularity Adjustment** - Recommends songs with similar popularity

### Features Analyzed
- Valence (mood/happiness)
- Acousticness
- Danceability
- Energy
- Instrumentalness
- Liveness
- Loudness
- Speechiness
- Popularity
- Duration
- Year

## Testing Script

To test the recommendation engine directly (without the API):

```powershell
python test_advanced_recommender.py
```

This will test recommendations for several different songs and show detailed cluster information.

## Troubleshooting

### Port already in use
If port 8000 is already in use, modify `api_fase3.py`:
```python
uvicorn.run(app, host="127.0.0.1", port=8001)  # Change to 8001 or any available port
```

### Module not found errors
Make sure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

### Dataset not found
Ensure you're running the API from the correct directory and that the datasets folder exists:
```
ML_project/
  datasets/
    data_metadata.csv
    data_features_scaled.csv
    data_filtered.csv
  scripts/
    api_fase3.py
    advanced_recommendation_engine.py
```

## Performance Notes

- **Initial startup**: 10-30 seconds (loading and training)
- **Recommendation time**: < 1 second per request
- **Dataset size**: 36,846 songs
- **Clusters**: 8 musical profiles
- **Algorithms per request**: 4 combined in ensemble

## Advanced Usage

### Customizing number of recommendations
The default is 5 recommendations. To change this, modify in `api_fase3.py`:
```python
recs = recommender.get_recommendations(idx, n_recommendations=10)  # Get 10 instead of 5
```

### Adjusting number of clusters
To change the number of musical profiles, modify in `api_fase3.py`:
```python
recommender = get_recommender_instance(
    metadata_path=str(METADATA_FILE),
    features_path=str(FEATURES_FILE),
    n_clusters=10  # Change from 8 to 10 clusters
)
```

### Adjusting algorithm weights
To modify how much each algorithm contributes, edit the `_ensemble_scores` method in `advanced_recommendation_engine.py`:
```python
weights = {
    'knn': 0.30,        # Adjust these weights
    'cosine': 0.25,     # They should sum to 1.0
    'weighted': 0.30,
    'popularity': 0.15
}
```

---

For more details, see `README_ADVANCED_SYSTEM.md`
