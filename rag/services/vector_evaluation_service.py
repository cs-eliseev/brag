from sklearn.metrics.pairwise import cosine_similarity

class VectorEvaluationService:
    @staticmethod
    def cosine(list1: list, list2: list) -> float:
        return cosine_similarity(list1, list2).flatten()