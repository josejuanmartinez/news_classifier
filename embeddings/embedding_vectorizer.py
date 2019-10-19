from gensim.models.doc2vec import Doc2Vec


class EmbeddingVectorizer:
    def __init__(self, file):
        self.model = Doc2Vec.load(file)

    def infer_vector(self, text, alpha, steps):
        return self.model.infer_vector(text, alpha=alpha, steps=steps)
