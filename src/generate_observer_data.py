import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Observer descriptions labeled by actual stage
observer_examples = {
    'stress': [
        "She seems really stressed lately, always rushing and forgetting things.",
        "He keeps saying he's fine but he looks exhausted after work every day.",
        "She's been working late every night this week and barely eats dinner.",
        "He snapped at me today over something small, which is unlike him.",
        "She seems overwhelmed with everything on her plate right now.",
        "He looks tired all the time but still shows up and gets things done.",
        "She keeps saying she just needs to get through this week.",
        "He seems distracted and a bit irritable but is still functioning okay.",
        "She forgot our plans twice this week, seems like a lot is on her mind.",
        "He looks run down but still laughs at jokes and engages in conversation.",
        "She mentioned she hasn't had a proper weekend in months.",
        "He seems stressed about work but still talks about future plans.",
        "She looks tired but she's still taking care of things around the house.",
        "He seems like he's carrying a lot but manages to keep going.",
        "She works really long hours and doesn't seem to have time for herself.",
        "He said he's just busy but I can tell it's more than that.",
        "She seems anxious about an upcoming deadline but is handling it.",
        "He looks like he needs a break but keeps pushing through.",
        "She mentioned having trouble sleeping but brushed it off quickly.",
        "He seems tense lately but still engages with the family at dinner.",
    ],
    'depression': [
        "She doesn't come out of her room much anymore.",
        "He used to love cricket but hasn't played in weeks.",
        "She stopped calling me like she used to.",
        "He barely speaks at dinner now, just stares at his phone.",
        "She used to love cooking but hasn't made anything in a long time.",
        "He lost interest in things he used to enjoy.",
        "She seems really flat lately, like the energy has gone out of her.",
        "He doesn't laugh at things that used to make him laugh.",
        "She's been sleeping a lot more than usual and skipping meals.",
        "He seems disconnected from everyone around him.",
        "She said she's fine but she doesn't seem like herself.",
        "He used to be so social but now he avoids going out.",
        "She looks empty, like she's just going through the motions.",
        "He stopped taking care of himself the way he used to.",
        "She hasn't talked about anything she's looking forward to in weeks.",
        "He seems like a different person from a few months ago.",
        "She barely responds to messages anymore, even from close friends.",
        "He used to have so many plans but stopped making any.",
        "She doesn't seem to find joy in anything lately.",
        "He looks like he's carrying something heavy and won't talk about it.",
    ],
    'crisis': [
        "She said she doesn't see the point of anything anymore.",
        "He told me he feels completely alone even when people are around.",
        "She mentioned that she wishes she could just disappear for a while.",
        "He said nothing is going to get better, no matter what he does.",
        "She broke down crying and said she's exhausted of fighting every day.",
        "He said he feels like a burden to everyone around him.",
        "She told me she hasn't felt anything in so long she's scared.",
        "He said he doesn't care what happens to him anymore.",
        "She mentioned she's been thinking about not being here.",
        "He said he feels completely hopeless and trapped.",
        "She told me she's been saying goodbye to people in her head.",
        "He said life feels meaningless and he doesn't know why he continues.",
        "She told me she's given up trying to get better.",
        "He broke down and said he can't do this anymore.",
        "She said she feels invisible and like no one would notice if she was gone.",
        "He told me he's been pushing everyone away on purpose.",
        "She said she's so tired of everything and just wants it to stop.",
        "He mentioned that he thinks everyone would be better off without him.",
        "She said she doesn't want to die but she doesn't want to live like this.",
        "He told me he feels nothing, completely numb, and it scares him.",
    ]
}

rows = []
for stage, texts in observer_examples.items():
    for text in texts:
        rows.append({'text': text, 'cascade_stage': stage, 'source': 'observer_synthetic'})

df = pd.DataFrame(rows)
print(f"Generated {len(df)} observer examples")
print(df['cascade_stage'].value_counts())

output_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'processed', 'observer_examples.csv'
)
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
