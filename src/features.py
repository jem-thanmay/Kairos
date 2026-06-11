import re
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def correct_spelling(text: str) -> str:
    """Fast spelling correction using common substitutions only."""
    import re
    # Common typo patterns - fast regex substitutions
    corrections = {
        r'\btierd\b': 'tired', r'\btierd\b': 'tired',
        r'\bfeelig\b': 'feeling', r'\bfeelign\b': 'feeling',
        r'\bdosent\b': 'doesn t', r'\bdoesnt\b': 'does not',
        r'\bisnt\b': 'is not', r'\bwont\b': 'will not',
        r'\bcant\b': 'cannot', r'\bdidnt\b': 'did not',
        r'\bhavent\b': 'have not', r'\bhasnt\b': 'has not',
        r'\bim\b': 'i am', r'\bive\b': 'i have',
        r'\bdont\b': 'do not', r'\bcouldnt\b': 'could not',
        r'\bwouldnt\b': 'would not', r'\bshouldnt\b': 'should not',
        r'\bseeems\b': 'seems', r'\bseemss\b': 'seems',
        r'\brealy\b': 'really', r'\breelly\b': 'really',
        r'\balwasy\b': 'always', r'\bawlays\b': 'always',
        r'\bexahsted\b': 'exhausted', r'\bexhausted\b': 'exhausted',
        r'\bdepresed\b': 'depressed', r'\bdepressd\b': 'depressed',
        r'\banxoius\b': 'anxious', r'\banxios\b': 'anxious',
        r'\blonley\b': 'lonely', r'\blonley\b': 'lonely',
        r'\bhopeles\b': 'hopeless', r'\bhopeles\b': 'hopeless',
    }
    text_lower = text.lower()
    for pattern, replacement in corrections.items():
        text_lower = re.sub(pattern, replacement, text_lower, flags=re.IGNORECASE)
    return text_lower

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

analyzer = SentimentIntensityAnalyzer()

ISOLATION_WORDS = {
    'alone', 'lonely', 'isolated', 'nobody', 'no one', 'empty',
    'disconnected', 'invisible', 'forgotten', 'abandoned', 'withdrawn',
    'distant', 'detached', 'numb', 'hollow'
}
HOPELESSNESS_WORDS = {
    'hopeless', 'pointless', 'worthless', 'useless', 'meaningless',
    'nothing', 'never', 'impossible', 'stuck', 'trapped', 'forever',
    'always', 'failure', 'failed', 'cant', 'cannot', 'give up',
    'no point', 'whats the point', 'tired of'
}
BURNOUT_WORDS = {
    'exhausted', 'drained', 'burnt out', 'burned out', 'overwhelmed',
    'overworked', 'no energy', 'falling behind', 'too much',
    'breaking point', 'running on empty', 'no motivation',
    'going through motions'
}
SELF_BLAME_WORDS = {
    'my fault', 'blame myself', 'i failed', 'i ruined', 'because of me',
    'i should have', 'i could have', 'hate myself', 'stupid',
    'pathetic', 'weak', 'worthless'
}
WITHDRAWAL_SIGNALS = {
    "doesn't come out", 'stopped going', "hasn't been", 'no longer',
    'used to', "doesn't talk", 'avoiding', 'isolating', 'withdrawn',
    'stopped eating', 'not eating', 'skipping meals', "hasn't eaten",
    'stopped sleeping', 'not sleeping', 'up all night', 'sleeping all day',
    'lost interest', "doesn't care", 'given up', 'stopped laughing',
    'not themselves', 'changed', 'different person', 'barely speaks'
}
FUNCTIONING_SIGNALS = {
    'still working', 'going to work', 'managing', 'getting through',
    'holding it together', 'functioning', 'keeping up', 'showing up',
    'fine at work', 'doing okay', 'seems okay', 'laughed', 'smiled'
}

def count_signals(text, word_set):
    t = text.lower()
    return sum(1 for w in word_set if w in t)

def extract_features(text: str) -> dict:
    text = str(text)
    words = text.lower().split()
    sentences = sent_tokenize(text)
    scores = analyzer.polarity_scores(text)

    first_person = sum(1 for w in words if w in {'i','me','my','myself','mine'})
    negations = sum(1 for w in words if w in {
        'not','no','never','nothing','nobody','nowhere',
        'neither','cant','wont','dont','doesnt','didnt',
        'isnt','arent','wasnt'
    })
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    return {
        'sentiment_neg': scores['neg'],
        'sentiment_neu': scores['neu'],
        'sentiment_pos': scores['pos'],
        'sentiment_compound': scores['compound'],
        'isolation_score': count_signals(text, ISOLATION_WORDS),
        'hopelessness_score': count_signals(text, HOPELESSNESS_WORDS),
        'burnout_score': count_signals(text, BURNOUT_WORDS),
        'self_blame_score': count_signals(text, SELF_BLAME_WORDS),
        'first_person_ratio': first_person / max(word_count, 1),
        'question_count': text.count('?'),
        'negation_ratio': negations / max(word_count, 1),
        'word_count': word_count,
        'avg_sentence_length': word_count / sentence_count,
        'exclamation_ratio': text.count('!') / sentence_count,
        'withdrawal_score': count_signals(text, WITHDRAWAL_SIGNALS),
        'functioning_score': count_signals(text, FUNCTIONING_SIGNALS)
    }


# Fix: normalize contractions before matching
_original_extract = extract_features

def extract_features(text: str, correct: bool = True) -> dict:
    text = str(text)
    if correct:
        text = correct_spelling(text)
    text = text.replace("hasn't", "has not").replace("haven't", "have not")
    text = text.replace("doesn't", "does not").replace("don't", "do not")
    text = text.replace("didn't", "did not").replace("won't", "will not")
    text = text.replace("can't", "cannot").replace("couldn't", "could not")
    text = text.replace("shouldn't", "should not").replace("wouldn't", "would not")
    text = text.replace("isn't", "is not").replace("aren't", "are not")
    text = text.replace("wasn't", "was not").replace("weren't", "were not")
    return _original_extract(text)
