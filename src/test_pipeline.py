import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_features
from src.predictor import predict

# Test 1 — observer describing someone else (burnout/stress)
text1 = "She hasn't eaten properly in three days. She laughed it off but sounded really flat. She keeps saying she's fine but I can hear how exhausted she is."

# Test 2 — observer describing someone withdrawing
text2 = "He doesn't come out of his room anymore. He used to love cricket but hasn't played in weeks. When I ask how he is he just says tired."

# Test 3 — first person crisis signals
text3 = "I don't see the point anymore. I'm exhausted and nothing I do matters. I feel completely alone even when people are around."

tests = [
    (text1, 'observer'),
    (text2, 'observer'),
    (text3, 'first_person'),
]

for i, (text, input_type) in enumerate(tests, 1):
    print(f"\n{'='*60}")
    print(f"TEST {i} [{input_type}]")
    print(f"Input: {text[:80]}...")
    features = extract_features(text)
    result = predict(features, input_type=input_type)
    print(f"Stage: {result['stage'].upper()}")
    print(f"Confidence: {result['confidence']}")
    print(f"Probabilities: {result['probabilities']}")
    print(f"\nGuidance: {result['guidance']['description']}")
    print(f"What to say: {result['guidance']['what_to_say']}")