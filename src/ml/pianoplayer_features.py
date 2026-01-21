"""
Extract pianoplayer physics-based features
Combines rule-based hand physics with our ML approach
"""

class SimplifiedHand:
    """
    Simplified version of pianoplayer's Hand class
    Extracts key physics metrics as features
    """
    
    # Finger speed weights (from pianoplayer/hand.py line 16)
    # Index 0 is dummy, fingers 1-5
    weights = [None, 1.1, 1.0, 1.1, 0.9, 0.8]
    
    # Black key penalty factors (from pianoplayer/hand.py line 17)
    # Thumb 50% slower on black, ring finger 80% speed
    bfactor = [None, 0.5, 1.0, 1.1, 0.8, 0.6]
    
    # Hand sizes in cm (from pianoplayer/hand.py setHandSize)
    hand_sizes = {
        'XXS': 15.0,  # 21.0 * 0.46
        'XS': 16.0,   # 21.0 * 0.64
        'S': 17.2,    # 21.0 * 0.82
        'M': 18.5,    # 21.0 * 1.0
        'L': 20.0,    # 21.0 * 1.1
        'XL': 21.5,   # 21.0 * 1.2
        'XXL': 23.0   # 21.0 * 1.3
    }
    
    def __init__(self, size='M'):
        self.max_span = self.hand_sizes.get(size, 18.5)
    
    def finger_speed_cost(self, from_finger, to_finger, interval, is_black):
        """
        Calculate velocity cost for finger transition
        Based on pianoplayer's ave_velocity function (hand.py line 98)
        """
        if from_finger == 0:  # First note
            return 0.0
        
        # Base finger weight
        weight = self.weights[to_finger]
        
        # Black key penalty
        if is_black:
            bfac = self.bfactor[to_finger]
        else:
            bfac = 1.0
        
        # Combined cost (higher = harder/slower)
        cost = 1.0 / (weight * bfac)
        
        # Add interval penalty for large jumps
        if abs(interval) > 7:  # More than a 5th
            cost *= 1.25
        
        return cost
    
    def is_forbidden_transition(self, f1, f2):
        """
        Check if transition violates pianoplayer's skip rules
        Based on hand.py skiprules() function (line 44)
        """
        if f1 == 0:
            return False
        
        # Forbidden finger crossings (non-thumb fingers)
        # From line 48: "skip if non-thumb fingers are crossings"
        forbidden = [
            (2, 3), (3, 2),  # Index-middle crossing
            (3, 4), (4, 3),  # Middle-ring crossing  
            (4, 5), (5, 4),  # Ring-pinkie
        ]
        
        return (f1, f2) in forbidden
    
    def hand_stretch_normalized(self, span_semitones):
        """
        Normalize hand stretch requirement
        Based on typical comfortable span of 12 semitones (octave)
        """
        return min(span_semitones / 12.0, 1.0)


def extract_pianoplayer_features(notes_df):
    """
    Add pianoplayer physics features to existing dataset
    """
    import pandas as pd
    import numpy as np
    
    hand = SimplifiedHand('M')
    
    # New physics-based features
    pp_speed_costs = []
    pp_stretches = []
    pp_forbidden = []
    pp_black_penalties = []
    
    for i, row in notes_df.iterrows():
        # Current note info
        curr_pitch = row.get('midi', 60)
        curr_finger = int(row.get('fingering', 3)) if pd.notna(row.get('fingering')) else 3
        is_black = row.get('is_black_key', False)
        
        # Previous note
        if i > 0:
            prev_row = notes_df.iloc[i-1]
            prev_pitch = prev_row.get('midi', 60)
            prev_finger = int(prev_row.get('fingering', 3)) if pd.notna(prev_row.get('fingering')) else 3
            interval = curr_pitch - prev_pitch
        else:
            prev_finger = 0
            interval = 0
        
        # Feature 1: Finger speed cost (pianoplayer's core metric)
        speed_cost = hand.finger_speed_cost(prev_finger, curr_finger, interval, is_black)
        pp_speed_costs.append(speed_cost)
        
        # Feature 2: Hand stretch (look ahead 3-5 notes)
        window_end = min(i + 5, len(notes_df))
        if window_end > i + 1:
            pitches = notes_df.iloc[i:window_end]['midi'].values
            span = max(pitches) - min(pitches)
            stretch = hand.hand_stretch_normalized(span)
        else:
            stretch = 0.0
        pp_stretches.append(stretch)
        
        # Feature 3: Forbidden transition
        forbidden = 1 if hand.is_forbidden_transition(prev_finger, curr_finger) else 0
        pp_forbidden.append(forbidden)
        
        # Feature 4: Black key penalty (especially for thumb)
        penalty = hand.bfactor[curr_finger] if is_black else 1.0
        pp_black_penalties.append(penalty)
    
    # Add to dataframe
    notes_df['pp_speed_cost'] = pp_speed_costs
    notes_df['pp_hand_stretch'] = pp_stretches
    notes_df['pp_forbidden'] = pp_forbidden
    notes_df['pp_black_penalty'] = pp_black_penalties
    
    print(f"✓ Added 4 pianoplayer physics features")
    
    return notes_df


if __name__ == '__main__':
    # Test
    import pandas as pd
    
    test_notes = pd.DataFrame({
        'midi': [60, 62, 64, 65, 67, 69, 71, 72],
        'fingering': [1, 2, 3, 1, 2, 3, 4, 5],
        'is_black_key': [False, False, False, False, False, False, False, False]
    })
    
    result = extract_pianoplayer_features(test_notes)
    print("\nPianoplayer Physics Features Test:")
    print(result[['midi', 'fingering', 'pp_speed_cost', 'pp_hand_stretch', 'pp_forbidden']])
