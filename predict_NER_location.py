import spacy
import re

import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_merchant_name(text):
    remainder = text
    # last two characters are country id
    country_id = remainder[len(text)-2] + remainder[len(text)-1]
    remainder = text[:len(text)-2].lower()

    return country_id, re.sub(r'[^a-zA-Z0-9\s]','',remainder).strip()

class NGramFuzzyMatcher:
    def __init__(self, reference_list, ngram_range=(2, 3)):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
        self.embeddings = self.vectorizer.fit_transform(reference_list)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.reference_list = reference_list

    def query(self, query, top_n = 1, threshold = 0.5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = np.where(similarities >= threshold)[0]
        top_indices = sorted(top_indices, key=lambda i: similarities[i], reverse=True)[:top_n]
        return [self.reference_list[x] for x in top_indices]

    def batch_query(self, query, top_n = 1, threshold = 0.5):
        query_vector = self.vectorizer.transform(query)
        similarities = cosine_similarity(query_vector, self.embeddings)
        top_indices = np.where(similarities >= threshold)

        list_indices = [0] * query_vector.shape[0]

        for i in range(len(list_indices)):
            indices_j = []
            for j in range(len(top_indices[0])):
                indices_j.append(top_indices[1][j])
            list_indices[i] = [self.reference_list[x] for x in sorted(indices_j, key=lambda n: similarities[i][n], reverse=True)[:top_n]]

        return list_indices


if __name__ == "__main__":
    # Load the saved model
    loaded_nlp = spacy.load("model/indonesian_location_ner_model")

    # load text enhancement
    file = open('data/ngram_fuzzy_matcher_class.pkl', 'rb')
    matcher = pickle.load(file)
    file.close()

    # Test on some text
    text = "TOKO ABADI JAYA JENDSUDBANYUMAS ID"
    country_id, text_clean = clean_merchant_name(text)
    doc = loaded_nlp(text_clean)

    # Print entities
    for ent in doc.ents:
        print(f"{country_id}: {ent.text} - {ent.label_}")

    # Print entities
    for ent in doc.ents:
        if ent.label_== "LOC":
            print("Augmented:", matcher.query(ent.text))
