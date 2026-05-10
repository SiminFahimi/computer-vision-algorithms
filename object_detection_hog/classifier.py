from .utils import cosine_similarity
def cosine_classifier(window_feat, prototypes):

    best_label = None
    best_score = -1

    for label, proto in prototypes.items():

        score = cosine_similarity(
            window_feat.flatten(),
            proto.flatten()
        )

        if score > best_score:
            best_score = score
            best_label = label
    
    return best_label, best_score