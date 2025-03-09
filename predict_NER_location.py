import spacy
# Load the saved model
loaded_nlp = spacy.load("model/indonesian_location_ner_model")

# Test on some text
def clean_merchant_name(text):
    remainder = text
    # last two characters are country id
    country_id = remainder[len(text)-2] + remainder[len(text)-1]
    remainder = text[:len(text)-2].lower()

    return re.sub(r'[^a-zA-Z0-9\s]','',remainder).strip()

text = "YASUI MBL BADUNG - BALIID"
text_clean = clean_merchant_name(text)
doc = loaded_nlp(text_clean)

# Print entities
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
