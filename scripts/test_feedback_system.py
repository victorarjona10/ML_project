"""
Test script for the Feedback System
====================================

This script demonstrates how the feedback system works:
1. Get recommendations for a song
2. Submit feedback (positive/negative) for recommendations
3. Get recommendations again and see how feedback affects results
"""

import requests
import time
from pathlib import Path

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def print_recommendations(response_data, title="Recommendations"):
    """Print recommendations in a formatted way"""
    print(f"\n{title}:")
    print("-" * 80)
    
    if "song_found" in response_data:
        song = response_data["song_found"]
        print(f"\nOriginal Song: {song['name']} - {song['artist']} ({song['year']})")
        print(f"Cluster: {song['cluster']['type']}")
        
        if response_data.get("algorithm_info", {}).get("feedback_applied"):
            print("✓ Feedback adjustments APPLIED")
        else:
            print("○ No feedback available (using base recommendations)")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(response_data.get("recommendations", []), 1):
        similarity = rec.get('similarity_percentage', rec.get('similarity_score', 0) * 100)
        print(f"\n  {i}. {rec['name']}")
        print(f"     Artist: {rec['artists']}")
        print(f"     Year: {rec['year']} | Popularity: {rec['popularity']}/100")
        print(f"     Similarity: {similarity:.2f}%")
        print(f"     Type: {rec.get('cluster_type', 'N/A')}")

def get_recommendations(song_name, artist_name=""):
    """Get recommendations from the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={"song_name": song_name, "artist_name": artist_name},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.json()}")
            return None
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API. Make sure it's running on http://127.0.0.1:8000")
        print("   Start it with: python api_fase3.py")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def submit_feedback(song_name, artist_name, recommended_song, recommended_artist, feedback_type):
    """Submit feedback for a recommendation"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json={
                "song_name": song_name,
                "artist_name": artist_name,
                "recommended_song": recommended_song,
                "recommended_artist": recommended_artist,
                "feedback_type": feedback_type
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            emoji = "👍" if feedback_type == "positive" else "👎"
            print(f"\n{emoji} Feedback '{feedback_type}' submitted for '{recommended_song}'")
            return True
        else:
            print(f"Error submitting feedback: {response.json()}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_feedback_stats():
    """Get feedback statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/feedback/stats", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Main test flow"""
    print_header("FEEDBACK SYSTEM DEMONSTRATION")
    print("\nThis script will:")
    print("1. Get recommendations for a song")
    print("2. Submit feedback (marking some as good, others as bad)")
    print("3. Get recommendations again to see the changes")
    
    # Test song (you can change this to any song in your dataset)
    TEST_SONG = "Shape of You"
    TEST_ARTIST = "Ed Sheeran"
    
    # Alternative songs to try:
    # TEST_SONG = "Despacito"
    # TEST_ARTIST = "Luis Fonsi"
    
    print(f"\nTest Song: '{TEST_SONG}' by {TEST_ARTIST}")
    input("\nPress Enter to start...")
    
    # Step 1: Get initial recommendations
    print_header("STEP 1: Initial Recommendations (No Feedback)")
    print("Getting recommendations for the first time...")
    
    initial_recs = get_recommendations(TEST_SONG, TEST_ARTIST)
    
    if not initial_recs:
        print("\n❌ Failed to get recommendations. Exiting.")
        return
    
    print_recommendations(initial_recs, "Initial Recommendations")
    
    # Save recommendation names for feedback
    recs_list = initial_recs.get("recommendations", [])
    
    if len(recs_list) < 3:
        print("\n❌ Not enough recommendations to test. Exiting.")
        return
    
    input("\nPress Enter to submit feedback...")
    
    # Step 2: Submit feedback
    print_header("STEP 2: Submitting Feedback")
    print("\nLet's mark some recommendations as good and others as bad:")
    
    # Mark first recommendation as POSITIVE
    print(f"\n→ Marking recommendation #1 as POSITIVE (good)")
    submit_feedback(
        TEST_SONG, TEST_ARTIST,
        recs_list[0]['name'], recs_list[0]['artists'],
        "positive"
    )
    
    # Mark second recommendation as NEGATIVE
    print(f"\n→ Marking recommendation #2 as NEGATIVE (bad)")
    submit_feedback(
        TEST_SONG, TEST_ARTIST,
        recs_list[1]['name'], recs_list[1]['artists'],
        "negative"
    )
    
    # If there's a third, mark it as NEGATIVE too
    if len(recs_list) >= 3:
        print(f"\n→ Marking recommendation #3 as NEGATIVE (bad)")
        submit_feedback(
            TEST_SONG, TEST_ARTIST,
            recs_list[2]['name'], recs_list[2]['artists'],
            "negative"
        )
    
    # Show feedback stats
    print("\n" + "-" * 80)
    print("Feedback Statistics:")
    stats_data = get_feedback_stats()
    if stats_data:
        stats = stats_data.get("statistics", {})
        print(f"  Total songs with feedback: {stats.get('total_songs_with_feedback', 0)}")
        print(f"  Total feedback entries: {stats.get('total_feedback_entries', 0)}")
        print(f"  Positive feedback: {stats.get('positive_feedback', 0)}")
        print(f"  Negative feedback: {stats.get('negative_feedback', 0)}")
    
    input("\nPress Enter to get new recommendations with feedback applied...")
    
    # Step 3: Get recommendations again with feedback
    print_header("STEP 3: Recommendations WITH Feedback Applied")
    print("Getting recommendations again...")
    print("Expected changes:")
    print("  - Recommendation #1 should rank HIGHER (positive feedback → +15% boost)")
    print("  - Recommendations #2 and #3 should rank LOWER (negative feedback → -30% penalty)")
    
    updated_recs = get_recommendations(TEST_SONG, TEST_ARTIST)
    
    if not updated_recs:
        print("\n❌ Failed to get updated recommendations.")
        return
    
    print_recommendations(updated_recs, "Updated Recommendations with Feedback")
    
    # Step 4: Compare results
    print_header("STEP 4: Comparison")
    print("\nBEFORE Feedback:")
    for i, rec in enumerate(recs_list[:5], 1):
        similarity = rec.get('similarity_percentage', rec.get('similarity_score', 0) * 100)
        print(f"  {i}. {rec['name'][:50]} - {similarity:.2f}%")
    
    print("\nAFTER Feedback:")
    for i, rec in enumerate(updated_recs.get("recommendations", [])[:5], 1):
        similarity = rec.get('similarity_percentage', rec.get('similarity_score', 0) * 100)
        print(f"  {i}. {rec['name'][:50]} - {similarity:.2f}%")
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print("1. Songs marked as POSITIVE should have higher similarity scores")
    print("2. Songs marked as NEGATIVE should have lower similarity scores")
    print("3. The ranking may change based on feedback adjustments")
    print("4. Feedback is persistent - it will affect future recommendations")
    print("\nFeedback file location: datasets/user_feedback.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
