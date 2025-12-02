import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from advanced_recommendation_engine import get_recommender_instance
from feedback_manager import get_feedback_manager

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
feedback_manager = None

def load_and_train():
    """
    Load and train the advanced multi-algorithm recommendation system
    """
    global recommender, feedback_manager
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
        
        # Initialize feedback manager
        feedback_manager = get_feedback_manager()
        
        print("=" * 60)
        print("[OK] Advanced Recommendation System Ready!")
        print("   - Multi-algorithm ensemble")
        print("   - Cluster-based specialization")
        print("   - Artist diversity enforcement")
        print("   - User feedback integration")
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

class FeedbackRequest(BaseModel):
    song_name: str
    artist_name: str
    recommended_song: str
    recommended_artist: str
    feedback_type: str  # "positive" or "negative" 

@app.on_event("startup")
def startup_event():
    load_and_train()

@app.get("/")
def home():
    return {
        "message": "Advanced Music Recommendation API v2.1 with Feedback",
        "status": "Active",
        "features": [
            "8 Musical Profile Clusters",
            "Multi-Algorithm Ensemble",
            "Artist Diversity",
            "Feature-Based Specialization",
            "User Feedback Learning"
        ],
        "endpoints": {
            "/recommend": "POST - Get song recommendations",
            "/feedback": "POST - Submit feedback on recommendations",
            "/feedback/stats": "GET - View feedback statistics"
        }
    }

@app.post("/recommend")
def recommend(request: SongRequest):
    """
    Advanced recommendation endpoint using multi-algorithm ensemble
    Automatically applies user feedback adjustments if available
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
        # Use advanced multi-algorithm recommendation WITH feedback integration
        recs = recommender.get_recommendations(
            idx, 
            n_recommendations=5,
            feedback_manager=feedback_manager  # Pass feedback manager
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

    # Check if feedback was applied
    feedback_applied = False
    if feedback_manager:
        feedback_data = feedback_manager.get_feedback_for_song(
            found_song['name'], 
            found_song['artists']
        )
        feedback_applied = feedback_data is not None

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
            "artist_diversity": True,
            "feedback_applied": feedback_applied
        }
    }

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a recommendation
    
    Args:
        song_name: Original song requested
        artist_name: Original artist requested
        recommended_song: Recommended song that was rated
        recommended_artist: Artist of recommended song
        feedback_type: "positive" or "negative"
    """
    if feedback_manager is None:
        raise HTTPException(status_code=503, detail="Feedback system not initialized")
    
    # Validate feedback_type
    if request.feedback_type not in ["positive", "negative"]:
        raise HTTPException(
            status_code=400, 
            detail="feedback_type must be 'positive' or 'negative'"
        )
    
    try:
        feedback_manager.add_feedback(
            song_name=request.song_name,
            artist_name=request.artist_name,
            recommended_song=request.recommended_song,
            recommended_artist=request.recommended_artist,
            feedback_type=request.feedback_type
        )
        
        return {
            "status": "success",
            "message": f"Feedback '{request.feedback_type}' registered for '{request.recommended_song}'",
            "original_song": request.song_name,
            "original_artist": request.artist_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")

@app.get("/feedback/stats")
def get_feedback_stats():
    """
    Get statistics about stored user feedback
    """
    if feedback_manager is None:
        raise HTTPException(status_code=503, detail="Feedback system not initialized")
    
    try:
        stats = feedback_manager.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)