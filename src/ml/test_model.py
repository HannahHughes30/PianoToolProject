"""Test the trained model with proper two-pass prediction"""
import pickle
from pathlib import Path

def note_to_midi(pitch):
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    if '#' in pitch:
        return (int(pitch[2]) + 1) * 12 + note_map[pitch[0]] + 1
    elif 'b' in pitch:
        return (int(pitch[2]) + 1) * 12 + note_map[pitch[0]] - 1
    return (int(pitch[1]) + 1) * 12 + note_map[pitch[0]]

def predict_fingering(notes, hand='right'):
    """Two-pass prediction to get better results"""
    model_path = Path('models/fingering_model.pkl')
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
        model = saved['model']
    
    print(f"\n{hand.upper()} HAND: {' → '.join(notes)}\n")
    
    # First pass: predict without next_fingering
    first_pass = []
    prev_f = 0
    
    for i, note in enumerate(notes):
        midi = note_to_midi(note)
        interval = midi - note_to_midi(notes[i-1]) if i > 0 else 0
        abs_interval = abs(interval)
        pos = i / len(notes)
        hand_enc = 0 if hand == 'right' else 1
        is_black = 1 if '#' in note or 'b' in note else 0
        is_step = 1 if abs_interval <= 2 else 0
        is_jump = 1 if abs_interval > 7 else 0
        is_high = 1 if midi > 72 else 0
        is_low = 1 if midi < 60 else 0
        is_asc = 1 if interval > 0 else 0
        is_desc = 1 if interval < 0 else 0
        
        features = [midi, interval, abs_interval, pos, hand_enc, prev_f, 0,
                   is_black, is_step, is_jump, is_high, is_low, is_asc, is_desc]
        
        finger = model.predict([features])[0]
        first_pass.append({
            'note': note,
            'midi': midi,
            'fingering': int(finger)
        })
        prev_f = finger
    
    # Second pass: re-predict with next_fingering
    results = []
    for i in range(len(notes)):
        note_data = first_pass[i]
        midi = note_data['midi']
        
        # Calculate features
        if i == 0:
            interval = 0
            prev_f = 0
        else:
            interval = midi - first_pass[i-1]['midi']
            prev_f = first_pass[i-1]['fingering']
        
        if i < len(notes) - 1:
            next_f = first_pass[i+1]['fingering']
        else:
            next_f = 0
        
        abs_interval = abs(interval)
        pos = i / len(notes)
        hand_enc = 0 if hand == 'right' else 1
        is_black = 1 if '#' in notes[i] or 'b' in notes[i] else 0
        is_step = 1 if abs_interval <= 2 else 0
        is_jump = 1 if abs_interval > 7 else 0
        is_high = 1 if midi > 72 else 0
        is_low = 1 if midi < 60 else 0
        is_asc = 1 if interval > 0 else 0
        is_desc = 1 if interval < 0 else 0
        
        features = [midi, interval, abs_interval, pos, hand_enc, prev_f, next_f,
                   is_black, is_step, is_jump, is_high, is_low, is_asc, is_desc]
        
        finger = model.predict([features])[0]
        results.append(int(finger))
        
        print(f"  {notes[i]:4s} → Finger {int(finger)}")
    
    return results

if __name__ == '__main__':
    print("="*60)
    print("FINGERING PREDICTION TEST")
    print("="*60)
    
    print("\nTest 1: C Major Scale")
    predict_fingering(['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'])
    
    print("\n" + "="*60)
    print("\nTest 2: G Major Scale")
    predict_fingering(['G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F#5', 'G5'])
    
    print("\n" + "="*60)
    print("\nTest 3: Descending C Major")
    predict_fingering(['C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4'])
    
    print("\n" + "="*60)
    print("\nTest 4: Arpeggios")
    predict_fingering(['C4', 'E4', 'G4', 'C5'])
    
    print("\n" + "="*60)
