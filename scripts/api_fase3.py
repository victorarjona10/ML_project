import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from advanced_recommendation_engine import get_recommender_instance

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================

# Ruta robusta usando __file__
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
METADATA_FILE = DATASETS_DIR / 'data_metadata.csv'
FEATURES_FILE = DATASETS_DIR / 'data_features_scaled.csv'

# ==========================================
# 2. LÓGICA DE MACHINE LEARNING AVANZADA
# ==========================================

recommender = None

def load_and_train():
    """
    Load and train the advanced multi-algorithm recommendation system
    """
    global recommender
    print("Initializing Advanced Recommendation Engine...")
    print("=" * 60)
    try:
        if not METADATA_FILE.exists() or not FEATURES_FILE.exists():
            print(f"[ERROR] No se encuentran archivos en {DATASETS_DIR}")
            return False

        # Initialize the advanced recommender with 8 musical clusters
        recommender = get_recommender_instance(
            metadata_path=str(METADATA_FILE),
            features_path=str(FEATURES_FILE),
            n_clusters=8  # 8 different musical profiles
        )
        
        print("=" * 60)
        print("[OK] Advanced Recommendation System Ready!")
        print("   - Multi-algorithm ensemble")
        print("   - Cluster-based specialization")
        print("   - Artist diversity enforcement")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"[ERROR] Error critico: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==========================================
# 3. DEFINICIÓN DE LA API
# ==========================================

app = FastAPI(
    title="Advanced Multi-Algorithm Music Recommendation API",
    description="""
    Sistema avanzado de recomendación musical que utiliza:
    - Clustering automático en 8 perfiles musicales
    - Algoritmos especializados por tipo de música
    - Ensemble de múltiples métodos (KNN, Cosine Similarity, Feature-Weighted Distance)
    - Ajuste por popularidad y diversidad de artistas
    
    Características analizadas: danceability, energy, acousticness, valence, 
    instrumentalness, speechiness, loudness, popularity, y más.
    """,
    version="2.0"
)

class SongRequest(BaseModel):
    song_name: str
    artist_name: str = "" 

@app.on_event("startup")
def startup_event():
    load_and_train()

@app.get("/")
def home():
    return {
        "message": "Advanced Music Recommendation API v2.0",
        "status": "Active",
        "features": [
            "8 Musical Profile Clusters",
            "Multi-Algorithm Ensemble",
            "Artist Diversity",
            "Feature-Based Specialization"
        ],
        "endpoint": "/recommend"
    }

@app.post("/recommend")
def recommend(request: SongRequest):
    """
    Advanced recommendation endpoint using multi-algorithm ensemble
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado.")

    print(f"[*] Buscando: '{request.song_name}' de '{request.artist_name}'")

    # Find song using the advanced recommender
    idx = recommender.find_song_index(request.song_name, request.artist_name)
    
    if idx is None:
        # Mensaje de error más descriptivo
        detail_msg = f"No se encontro la cancion '{request.song_name}'"
        if request.artist_name:
            detail_msg += f" del artista '{request.artist_name}'"
        
        raise HTTPException(status_code=404, detail=detail_msg)

    # Get song info
    found_song = recommender.df_metadata.iloc[idx]
    found_features = recommender.df_features.iloc[idx]
    cluster_id = recommender.get_song_cluster(idx)
    cluster_info = recommender.cluster_profiles[cluster_id]
    
    try:
        # Use advanced multi-algorithm recommendation
        recs = recommender.get_recommendations(idx, n_recommendations=5)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

    return {
        "song_found": {
            "name": found_song['name'],
            "artist": found_song['artists'],
            "year": int(found_song['year']),
            "popularity": int(found_features['popularity']),
            "audio_features": {
                "danceability": round(float(found_features['danceability']), 3),
                "energy": round(float(found_features['energy']), 3),
                "valence": round(float(found_features['valence']), 3),
                "acousticness": round(float(found_features['acousticness']), 3),
                "speechiness": round(float(found_features['speechiness']), 3),
            },
            "cluster": {
                "id": int(cluster_id),
                "type": cluster_info['type'],
                "key_features": cluster_info['key_features']
            }
        },
        "recommendations": recs,
        "algorithm_info": {
            "method": "Multi-Algorithm Ensemble",
            "algorithms_used": [
                "K-Nearest Neighbors (KNN)",
                "Cosine Similarity",
                "Feature-Weighted Distance",
                "Popularity Adjustment"
            ],
            "cluster_based": True,
            "artist_diversity": True
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)