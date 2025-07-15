import difflib

def correct_term(term, vocabulary):
    """
    Returns the closest match from vocabulary for a given term based on similarity.
    If no close match is found, returns the original term.
    """
    matches = difflib.get_close_matches(term, vocabulary, n=1, cutoff=0.8)
    return matches[0] if matches else term

def correct_query(query, vocabulary):
    """
    Corrects each word in the query based on the closest match in the vocabulary.
    """
    #return ' '.join([correct_term(word, vocabulary) for word in query.split()])
    return [correct_term(word, vocabulary) for word in query]