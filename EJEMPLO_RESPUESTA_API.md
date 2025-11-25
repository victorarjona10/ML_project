# Ejemplo de Respuesta Mejorada de la API

## Ahora la respuesta incluye:

### 1. Canción Encontrada
```json
{
  "song_found": {
    "name": "Waka Waka (This Time for Africa) [The Official 2010 FIFA World Cup (TM) Song]",
    "artist": "['Shakira', 'Freshlyground']",
    "year": 2010,
    "popularity": 78,  // ← POPULARIDAD de la canción (0-100)
    "audio_features": {
      "danceability": 0.823,
      "energy": 0.751,
      "valence": 0.892,
      "acousticness": 0.041,
      "speechiness": 0.068
    },
    "cluster": {
      "id": 3,
      "type": "Electronic/Dance/Pop",
      "key_features": ["danceability", "energy", "valence", "loudness", "popularity"]
    }
  }
}
```

### 2. Recomendaciones
```json
{
  "recommendations": [
    {
      "name": "Canción Recomendada 1",
      "artists": "['Artista']",
      "year": 2011,
      "popularity": 75,  // ← POPULARIDAD de la canción recomendada (0-100)
      "similarity_score": 0.8456,  // ← SIMILITUD (0-1) - Más cercano a 1 = Más similar
      "similarity_percentage": 84.56,  // ← SIMILITUD EN PORCENTAJE (0-100%)
      "cluster_type": "Electronic/Dance/Pop"
    },
    {
      "name": "Canción Recomendada 2",
      "artists": "['Artista 2']",
      "year": 2012,
      "popularity": 82,
      "similarity_score": 0.8234,  // ← Esta es menos similar (82.34%)
      "similarity_percentage": 82.34,
      "cluster_type": "Electronic/Dance/Pop"
    }
  ]
}
```

## Diferencias Importantes:

| Campo | Qué Significa | Rango |
|-------|---------------|-------|
| **popularity** | Qué tan popular es la canción en general | 0-100 |
| **similarity_score** | Qué tan similar es a la canción de entrada | 0-1 |
| **similarity_percentage** | Lo mismo pero en porcentaje | 0-100% |

## Interpretación:

- **similarity_percentage cerca de 100%** = Muy similar a tu canción
- **similarity_percentage cerca de 70-80%** = Similar pero con algunas diferencias
- **similarity_percentage cerca de 50%** = Algo similar

- **popularity cerca de 100** = Canción muy popular/conocida
- **popularity cerca de 50** = Popularidad media
- **popularity cerca de 0** = Canción poco conocida

## Ejemplo Real:

Si buscas "Waka Waka" de Shakira:
- La canción tiene `popularity: 78` (bastante popular)
- Las recomendaciones tendrán:
  - `similarity_percentage`: 85% → Muy parecida musicalmente
  - `popularity`: 70-85 → De popularidad similar
  - `cluster_type`: "Electronic/Dance/Pop" → Del mismo estilo musical

---

**Reinicia el servidor** para ver estos cambios:
```powershell
python api_fase3.py
```

Luego prueba de nuevo en: http://127.0.0.1:8000/docs
