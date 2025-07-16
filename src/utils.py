import math

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two sparse TF-IDF vectors.

    Parameters:
        vec1 (dict): First vector (e.g., query vector), with terms as keys and weights as values.
        vec2 (dict): Second vector (e.g., document vector), in the same format.

    Returns:
        float: Cosine similarity score between the two vectors.
               Returns 0.0 if either vector has zero magnitude.
    """
    # Calculate dot product between the two vectors
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1)
    # Calculate the Euclidean norm (magnitude) of each vector
    norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    # Avoid division by zero if either vector has zero magnitude
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # Return cosine similarity score
    return dot_product / (norm1 * norm2)