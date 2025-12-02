# Sistema de Feedback para Recomendaciones Musicales

## 📋 Resumen

Se ha implementado un **sistema de feedback de usuario** que permite mejorar las recomendaciones musicales basándose en la retroalimentación del usuario sobre canciones recomendadas previamente.

## 🎯 Funcionalidad

### ¿Qué hace?

1. **Captura feedback del usuario**: Después de recibir recomendaciones, el usuario puede marcar cada canción como:
   - ✅ **Positiva** (le gustó la recomendación)
   - ❌ **Negativa** (no le gustó la recomendación)

2. **Almacena el feedback**: Se guarda en un archivo JSON persistente (`datasets/user_feedback.json`)

3. **Aplica ajustes automáticos**: Cuando el usuario pide recomendaciones de nuevo para la misma canción:
   - Canciones marcadas como **positivas** reciben un **boost de +15%** en su score de similitud
   - Canciones marcadas como **negativas** reciben una **penalización de -30%** en su score

4. **Re-rankea resultados**: Las recomendaciones se reordenan basándose en los scores ajustados

## 🏗️ Arquitectura

### Archivos Nuevos

1. **`scripts/feedback_manager.py`** (340 líneas)
   - Clase `FeedbackManager`: Maneja almacenamiento y recuperación de feedback
   - Métodos principales:
     - `add_feedback()`: Guarda feedback de usuario
     - `get_feedback_adjustment()`: Devuelve multiplicador de ajuste (0.70, 1.0, o 1.15)
     - `get_statistics()`: Estadísticas de feedback almacenado

2. **`scripts/test_feedback_system.py`** (240 líneas)
   - Script de prueba interactivo
   - Demuestra el flujo completo: recomendar → feedback → re-recomendar

### Archivos Modificados

1. **`scripts/advanced_recommendation_engine.py`**
   - Método `get_recommendations()` actualizado para aceptar `feedback_manager` opcional
   - Aplica ajustes de feedback antes de devolver resultados

2. **`scripts/api_fase3.py`**
   - Nuevo endpoint `POST /feedback`: Enviar feedback
   - Nuevo endpoint `GET /feedback/stats`: Ver estadísticas
   - Endpoint `/recommend` actualizado para usar feedback automáticamente
   - Campo `feedback_applied` en respuesta

## 📊 Estructura de Datos (JSON)

```json
{
  "shape of you|ed sheeran": {
    "song_name": "Shape of You",
    "artist_name": "Ed Sheeran",
    "recommendations": [
      {
        "name": "Perfect",
        "artist": "Ed Sheeran",
        "feedback": "positive",
        "timestamp": "2025-12-02T10:30:45"
      },
      {
        "name": "Thinking Out Loud",
        "artist": "Ed Sheeran",
        "feedback": "negative",
        "timestamp": "2025-12-02T10:31:12"
      }
    ]
  }
}
```

## 🔧 Cómo Usar

### 1. Obtener Recomendaciones

```bash
POST http://127.0.0.1:8000/recommend
{
  "song_name": "Shape of You",
  "artist_name": "Ed Sheeran"
}
```

**Respuesta incluye**: `algorithm_info.feedback_applied` (true/false)

### 2. Enviar Feedback

```bash
POST http://127.0.0.1:8000/feedback
{
  "song_name": "Shape of You",
  "artist_name": "Ed Sheeran",
  "recommended_song": "Perfect",
  "recommended_artist": "Ed Sheeran",
  "feedback_type": "positive"
}
```

### 3. Ver Estadísticas

```bash
GET http://127.0.0.1:8000/feedback/stats
```

**Respuesta**:
```json
{
  "status": "success",
  "statistics": {
    "total_songs_with_feedback": 5,
    "total_feedback_entries": 23,
    "positive_feedback": 15,
    "negative_feedback": 8,
    "positive_ratio": 0.652
  }
}
```

### 4. Probar el Sistema Completo

```bash
# Asegúrate de que la API esté corriendo
python scripts/api_fase3.py

# En otra terminal, ejecuta el script de prueba
python scripts/test_feedback_system.py
```

## 🧮 Algoritmo de Ajuste

### Flujo de Recomendación CON Feedback

1. **Entrada**: Usuario pide recomendaciones para "Shape of You"
2. **Algoritmo base**: Sistema genera 10 candidatos con scores (0-1)
3. **Lookup de feedback**: Busca feedback previo para "Shape of You"
4. **Aplicar ajustes**:
   ```python
   for cada recomendación:
       if tiene feedback positivo:
           score_ajustado = score_original * 1.15  # +15%
       elif tiene feedback negativo:
           score_ajustado = score_original * 0.70  # -30%
       else:
           score_ajustado = score_original  # sin cambio
   ```
5. **Re-rankeo**: Ordenar por `score_ajustado` (descendente)
6. **Resultado**: Top 5 recomendaciones ajustadas

### Ejemplo Numérico

```
ANTES del feedback:
1. "Perfect" - 0.92 (92%)
2. "Castle on the Hill" - 0.89 (89%)
3. "Thinking Out Loud" - 0.87 (87%)

Usuario marca:
- "Perfect" → POSITIVO
- "Thinking Out Loud" → NEGATIVO

DESPUÉS del feedback:
1. "Perfect" - 1.058 (105.8%) ← subió por +15%
2. "Castle on the Hill" - 0.89 (89%) ← sin cambio
3. "Thinking Out Loud" - 0.609 (60.9%) ← bajó por -30%
```

## 📈 Ventajas

1. **Personalización**: Recomendaciones mejoran con el uso
2. **Persistencia**: Feedback se guarda entre sesiones
3. **Transparencia**: Usuario ve si feedback fue aplicado (`feedback_applied: true`)
4. **Sin re-entrenamiento**: Ajustes en tiempo real, sin necesidad de re-entrenar modelos
5. **Retrocompatible**: Si no hay feedback, funciona como antes

## ⚠️ Limitaciones Actuales

1. **No aprende patrones globales**: Solo ajusta canciones específicas que recibieron feedback
2. **No hay decay temporal**: Feedback viejo tiene mismo peso que reciente
3. **No considera contexto**: No diferencia feedback según hora del día, mood, etc.
4. **Almacenamiento local**: Archivo JSON (no base de datos escalable)
5. **Sin usuario multi-tenant**: Un solo archivo de feedback compartido

## 🚀 Mejoras Futuras Posibles

1. **Decay temporal**: Feedback reciente pesa más que antiguo
2. **Aprendizaje de patrones**: 
   - Si usuario rechaza muchas canciones de un cluster → bajar peso de ese cluster
   - Si acepta artistas similares → boost para artistas parecidos
3. **Feedback implícito**: Tiempo de escucha, skips, replays
4. **Multi-usuario**: Separar feedback por usuario_id
5. **Base de datos**: PostgreSQL/MongoDB en vez de JSON
6. **Validación cruzada**: Métricas offline (precision@k después de feedback)
7. **A/B testing**: Comparar versión con/sin feedback

## 🔬 Testing

### Test Manual Rápido

```bash
# Terminal 1: Iniciar API
python scripts/api_fase3.py

# Terminal 2: Probar con curl o test script
python scripts/test_feedback_system.py
```

### Verificar Archivo de Feedback

```bash
cat datasets/user_feedback.json
# o en Windows:
type datasets\user_feedback.json
```

## 📝 Notas de Implementación

- **Normalización de claves**: Song/Artist se normalizan a lowercase para matching
- **Sobrescritura**: Feedback nuevo para misma canción sobrescribe el anterior
- **Thread-safe**: ⚠️ No implementado (usar locks si múltiples workers)
- **Encoding**: UTF-8 para soportar caracteres especiales en nombres

## 🎓 Para Presentación

### Elevator Pitch (30 segundos)

"Implementamos un sistema de feedback que aprende de las preferencias del usuario. Cuando marcas recomendaciones como buenas o malas, el sistema ajusta automáticamente los scores (+15% para positivas, -30% para negativas) en futuras recomendaciones de la misma canción. Todo se guarda en un archivo JSON persistente."

### Demo Steps (3 minutos)

1. Mostrar recomendación inicial para una canción
2. Marcar 2-3 recomendaciones (algunas positivas, otras negativas)
3. Pedir recomendaciones de nuevo
4. Señalar cómo el ranking cambió basándose en feedback
5. Mostrar endpoint `/feedback/stats`

### Bullets Técnicos

- ✅ Almacenamiento persistente en JSON
- ✅ Ajuste automático de scores (±15% / -30%)
- ✅ 3 endpoints nuevos: `/feedback`, `/feedback/stats`, actualizado `/recommend`
- ✅ Backward compatible (funciona sin feedback)
- ✅ Script de test interactivo incluido

---

**Versión del Sistema**: 2.1  
**Fecha de Implementación**: Diciembre 2025  
**Archivos Modificados**: 2  
**Archivos Nuevos**: 2  
**Líneas de Código Añadidas**: ~580
