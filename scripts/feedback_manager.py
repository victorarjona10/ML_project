"""
Feedback Manager for Music Recommendation System
=================================================

This module handles user feedback storage and retrieval to improve recommendations.
Stores feedback in JSON format with structure:
{
    "song_name|artist_name": {
        "song_name": str,
        "artist_name": str,
        "recommendations": [
            {
                "name": str,
                "artist": str,
                "feedback": "positive" | "negative",
                "timestamp": str
            }
        ]
    }
}
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Literal


class FeedbackManager:
    """Manages user feedback for music recommendations"""
    
    def __init__(self, feedback_file: str = None):
        """
        Initialize the feedback manager
        
        Args:
            feedback_file: Path to the JSON file storing feedback data
        """
        if feedback_file is None:
            # Default to datasets folder
            feedback_file = Path(__file__).parent.parent / "datasets" / "user_feedback.json"
        
        self.feedback_file = Path(feedback_file)
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict:
        """Load feedback data from JSON file"""
        if not self.feedback_file.exists():
            print(f"[*] Creating new feedback file at {self.feedback_file}")
            return {}
        
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[OK] Loaded feedback data: {len(data)} songs with feedback")
                return data
        except json.JSONDecodeError:
            print(f"[!] Warning: Corrupted feedback file, starting fresh")
            return {}
        except Exception as e:
            print(f"[ERROR] Failed to load feedback: {e}")
            return {}
    
    def _save_feedback(self):
        """Save feedback data to JSON file"""
        try:
            # Ensure directory exists
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
            
            print(f"[OK] Feedback saved to {self.feedback_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save feedback: {e}")
    
    def _get_key(self, song_name: str, artist_name: str) -> str:
        """Generate a unique key for a song"""
        # Normalize to lowercase and strip whitespace
        song = song_name.lower().strip()
        artist = artist_name.lower().strip()
        return f"{song}|{artist}"
    
    def add_feedback(
        self,
        song_name: str,
        artist_name: str,
        recommended_song: str,
        recommended_artist: str,
        feedback_type: Literal["positive", "negative"]
    ):
        """
        Add user feedback for a recommendation
        
        Args:
            song_name: Original song requested by user
            artist_name: Original artist requested by user
            recommended_song: Name of recommended song
            recommended_artist: Artist of recommended song
            feedback_type: "positive" if user liked it, "negative" if not
        """
        key = self._get_key(song_name, artist_name)
        
        # Initialize song entry if it doesn't exist
        if key not in self.feedback_data:
            self.feedback_data[key] = {
                "song_name": song_name,
                "artist_name": artist_name,
                "recommendations": []
            }
        
        # Add feedback entry
        feedback_entry = {
            "name": recommended_song,
            "artist": recommended_artist,
            "feedback": feedback_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if feedback already exists for this recommendation
        recommendations = self.feedback_data[key]["recommendations"]
        existing_idx = None
        
        for idx, rec in enumerate(recommendations):
            if (rec["name"].lower() == recommended_song.lower() and 
                rec["artist"].lower() == recommended_artist.lower()):
                existing_idx = idx
                break
        
        if existing_idx is not None:
            # Update existing feedback
            recommendations[existing_idx] = feedback_entry
            print(f"[*] Updated feedback for '{recommended_song}' by {recommended_artist}")
        else:
            # Add new feedback
            recommendations.append(feedback_entry)
            print(f"[*] Added {feedback_type} feedback for '{recommended_song}' by {recommended_artist}")
        
        self._save_feedback()
    
    def get_feedback_for_song(
        self,
        song_name: str,
        artist_name: str
    ) -> Optional[Dict]:
        """
        Get all feedback for a specific song
        
        Args:
            song_name: Song name
            artist_name: Artist name
            
        Returns:
            Dictionary with feedback data or None if no feedback exists
        """
        key = self._get_key(song_name, artist_name)
        return self.feedback_data.get(key)
    
    def get_feedback_adjustment(
        self,
        song_name: str,
        artist_name: str,
        recommended_song: str,
        recommended_artist: str
    ) -> float:
        """
        Get the adjustment factor for a recommendation based on past feedback
        
        Args:
            song_name: Original song requested
            artist_name: Original artist requested
            recommended_song: Name of recommended song
            recommended_artist: Artist of recommended song
            
        Returns:
            Adjustment multiplier (1.0 = no change, >1.0 = boost, <1.0 = penalty)
        """
        feedback_data = self.get_feedback_for_song(song_name, artist_name)
        
        if not feedback_data:
            return 1.0  # No feedback, no adjustment
        
        # Look for feedback on this specific recommendation
        for rec in feedback_data["recommendations"]:
            if (rec["name"].lower() == recommended_song.lower() and 
                rec["artist"].lower() == recommended_artist.lower()):
                
                if rec["feedback"] == "positive":
                    return 1.15  # Boost by 15%
                elif rec["feedback"] == "negative":
                    return 0.70  # Penalty of 30%
        
        return 1.0  # No feedback for this specific recommendation
    
    def get_all_feedback_adjustments(
        self,
        song_name: str,
        artist_name: str
    ) -> Dict[str, float]:
        """
        Get all feedback adjustments for a song as a dictionary
        
        Args:
            song_name: Original song
            artist_name: Original artist
            
        Returns:
            Dictionary mapping "song_name|artist_name" to adjustment factor
        """
        feedback_data = self.get_feedback_for_song(song_name, artist_name)
        
        if not feedback_data:
            return {}
        
        adjustments = {}
        for rec in feedback_data["recommendations"]:
            key = self._get_key(rec["name"], rec["artist"])
            
            if rec["feedback"] == "positive":
                adjustments[key] = 1.15
            elif rec["feedback"] == "negative":
                adjustments[key] = 0.70
        
        return adjustments
    
    def get_statistics(self) -> Dict:
        """Get statistics about stored feedback"""
        total_songs = len(self.feedback_data)
        total_feedback = sum(len(data["recommendations"]) for data in self.feedback_data.values())
        
        positive_count = 0
        negative_count = 0
        
        for data in self.feedback_data.values():
            for rec in data["recommendations"]:
                if rec["feedback"] == "positive":
                    positive_count += 1
                elif rec["feedback"] == "negative":
                    negative_count += 1
        
        return {
            "total_songs_with_feedback": total_songs,
            "total_feedback_entries": total_feedback,
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "positive_ratio": positive_count / total_feedback if total_feedback > 0 else 0
        }
    
    def clear_feedback_for_song(self, song_name: str, artist_name: str):
        """Clear all feedback for a specific song"""
        key = self._get_key(song_name, artist_name)
        if key in self.feedback_data:
            del self.feedback_data[key]
            self._save_feedback()
            print(f"[*] Cleared feedback for '{song_name}' by {artist_name}")
    
    def clear_all_feedback(self):
        """Clear all feedback data"""
        self.feedback_data = {}
        self._save_feedback()
        print(f"[*] Cleared all feedback data")


# Singleton instance for API usage
_feedback_manager_instance = None

def get_feedback_manager(feedback_file: str = None):
    """Get or create the feedback manager instance"""
    global _feedback_manager_instance
    
    if _feedback_manager_instance is None:
        _feedback_manager_instance = FeedbackManager(feedback_file)
    
    return _feedback_manager_instance
