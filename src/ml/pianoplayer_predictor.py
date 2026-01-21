"""
Run pianoplayer's full algorithm to get expert predictions
These become features for our ensemble model
"""
import sys
sys.path.append('pianoplayer')
from hand import Hand
import pandas as pd
import numpy as np

class PianoplayerPredictor:
    """
    Wrapper around pianoplayer's optimize_seq algorithm
    """
    
    def __init__(self, hand_size='M'):
        self.hand = Hand("right", hand_size)
        
    def predict_fingering_window(self, notes_window, start_finger=0):
        """
        Use pianoplayer's combinatorial search to predict fingering
        
        Args:
            notes_window: List of 8-9 note dicts with 'midi', 'is_black_key'
            start_finger: Previously used finger (0 if first note)
            
        Returns:
            (predicted_finger, confidence_score)
        """
        if len(notes_window) < 5:
            return (3, 0.0)  # Default to middle finger, low confidence
        
        # Create mock Note objects for pianoplayer
        class MockNote:
            def __init__(self, midi_pitch, is_black, time_offset, duration=1.0):
                self.x = self._midi_to_position(midi_pitch)
                self.time = time_offset
                self.duration = duration
                self.isBlack = is_black
                self.isChord = False
                self.measure = 1
                
            def _midi_to_position(self, midi):
                # Convert MIDI to keyboard position (cm)
                # C4 (60) = 0, each semitone = ~1.65cm
                return (midi - 60) * 1.65
        
        # Convert to MockNotes
        mock_notes = []
        for i, note in enumerate(notes_window[:9]):
            mock_note = MockNote(
                note.get('midi', 60),
                note.get('is_black_key', False),
                i * 0.5,  # Mock time
                1.0
            )
            mock_notes.append(mock_note)
        
        # Pad to 9 notes if needed
        while len(mock_notes) < 9:
            last = mock_notes[-1]
            mock_notes.append(MockNote(last.x, last.isBlack, len(mock_notes) * 0.5))
        
        # Run pianoplayer's optimization
        try:
            fingering_sequence, cost = self.hand.optimize_seq(mock_notes, start_finger)
            
            predicted_finger = fingering_sequence[0]
            
            # Confidence based on cost (lower cost = higher confidence)
            # Normalize cost to 0-1 confidence
            confidence = max(0, min(1, 1.0 - (cost / 100.0)))
            
            return (predicted_finger, confidence)
            
        except Exception as e:
            # If optimization fails, return default
            return (3, 0.0)


def add_pianoplayer_predictions(df):
    """
    Add pianoplayer's predictions as features to dataset
    """
    print("\n⏳ Running pianoplayer algorithm on all notes...")
    print("   (This may take a minute - combinatorial search)")
    
    predictor = PianoplayerPredictor('M')
    
    pp_predictions = []
    pp_confidences = []
    
    for i in range(len(df)):
        # Get window of next 8 notes
        window_end = min(i + 9, len(df))
        notes_window = df.iloc[i:window_end].to_dict('records')
        
        # Get previous finger
        if i > 0:
            prev_finger = int(df.iloc[i-1].get('fingering', 0)) if pd.notna(df.iloc[i-1].get('fingering')) else 0
        else:
            prev_finger = 0
        
        # Get pianoplayer's prediction
        pred_finger, confidence = predictor.predict_fingering_window(notes_window, prev_finger)
        
        pp_predictions.append(pred_finger)
        pp_confidences.append(confidence)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(df)} notes...")
    
    df['pp_prediction'] = pp_predictions
    df['pp_confidence'] = pp_confidences
    
    print(f"\n✓ Added pianoplayer predictions")
    
    return df


if __name__ == '__main__':
    # Test
    test_notes = pd.DataFrame({
        'midi': [60, 62, 64, 65, 67, 69, 71, 72],
        'fingering': [1, 2, 3, 1, 2, 3, 4, 5],
        'is_black_key': [False] * 8
    })
    
    result = add_pianoplayer_predictions(test_notes)
    print("\nPianoplayer Predictions:")
    print(result[['midi', 'fingering', 'pp_prediction', 'pp_confidence']])
