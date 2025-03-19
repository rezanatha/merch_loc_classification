import spacy
import re
# Load the saved model
loaded_nlp = spacy.load("model/indonesian_location_ner_model")

# Test on some text
def clean_merchant_name(text):
    remainder = text
    # last two characters are country id
    country_id = remainder[len(text)-2] + remainder[len(text)-1]
    remainder = text[:len(text)-2].lower()

    return country_id, re.sub(r'[^a-zA-Z0-9\s]','',remainder).strip()

text = "6B40 LAWSON JENDRAL SUDIRBANYUMAS ID"
country_id, text_clean = clean_merchant_name(text)
doc = loaded_nlp(text_clean)

# Print entities
for ent in doc.ents:
    print(f"{country_id}: {ent.text} - {ent.label_}")

# text augmentation
import pickle
file = open('data/city_dictionary.pkl', 'rb')
city_dictionary = pickle.load(file)
file.close()

def jaccard_similarity(ngrams1, ngrams2):
    _ngrams1 = set(ngrams1)
    _ngrams2 = set(ngrams2)

    intersection = len(_ngrams1.intersection(_ngrams2))
    union = len(_ngrams1.union(_ngrams2))

    return intersection / union

def enhance_location_from_dict(row, cba_threshold = 0.6, threshold = 0.8, all_text_threshold = 0.7):
    if row is None or len(row) == 0:
        return None

    # SPECIAL CASE: BADUNG & BANDUNG
    if re.search(r'(bandung\s?barat)', row):
        return row, [(1.1, 'bandung barat')]
    if re.search(r'(bdg\s?barat)', row):
        return row, [(1.1, 'bdg barat')]
    if re.search(r'bandung', row):
        return row, [(1.1, 'bandung')]
    if re.search(r'(bdg)', row):
        return row, [(1.1, 'bdg')]
    if re.search(r'badung', row):
        return row, [(1.1, 'badung')]

    #char by all
    match = []
    for loc_char in row:
        if loc_char not in city_dictionary['full_text']:
            continue

        for possible_loc in city_dictionary['full_text'][loc_char]:
            score = jaccard_similarity(possible_loc, row)
            if possible_loc in row:
                score += 0.01

            if score >= cba_threshold:
                match.append((score, possible_loc, 'cba'))

    # word by word

    for loc_word in row.split(" "):
        if len(loc_word) == 0:
            continue
        if loc_word[0] not in city_dictionary['full_text']:
            continue

        for possible_loc in city_dictionary['full_text'][loc_word[0]]:
            score = jaccard_similarity(possible_loc, loc_word)
            if possible_loc in loc_word:
                score += 0.01

            if score >= threshold:
                match.append((score,possible_loc, 'wbw')) # change loc_word to possible_loc for full enhancement
    # all by all
    if row[0] in city_dictionary['full_text']:
        for possible_loc in city_dictionary['full_text'][row[0]]:
            score = jaccard_similarity(possible_loc, row)
            if score >= all_text_threshold:
                match.append((1.01*score, possible_loc ,'aba')) # change row to possible_loc for full enhancement

    return row, sorted(match, key=lambda x: x[0], reverse=True)

def extract_enhanced_location(row):
    enhanced = enhance_location_from_dict(row)
    print(enhanced)

    if len(enhanced[1]) == 0:
        return None
    if len(enhanced[1][0]) == 0:
        return None

    return enhanced[1][0][1]

# Print entities
for ent in doc.ents:
    if ent.label_== "LOC":
        print("Augmented:", extract_enhanced_location(ent.text))
