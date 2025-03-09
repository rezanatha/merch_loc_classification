import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

import polars as pl
import random


train = pl.read_parquet('data/merchant_ner_labeled_train.parquet')

def create_spacy_dataset(data):
    spacy_data = []
    for row in data.iter_rows():
        spacy_data.append((row[1], {"entities": [(row[5][0], row[5][1], "LOC")]}))

    return spacy_data

spacy_data = create_spacy_data(train)
random.shuffle(spacy_data)
train_data = spacy_data[:int(len(spacy_data) * 0.8)]
test_data = spacy_data[int(len(spacy_data) * 0.8):]

def create_model(train_data):
    nlp = spacy.blank("id")  # create blank model (adjust language as needed)
    print("Created blank model")

    # Add NER component if it doesn't exist
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add entity labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    return nlp


def convert_to_spacy(data, output_path):
    db = DocBin()
    nlp_model = spacy.blank("id")

    for text, annots in data:
        doc = nlp_model.make_doc(text)
        ents = []
        for start, end, label in annots.get("entities"):
            span = doc.char_span(start, end, label=label)

            if span is not None:
                ents.append(span)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        db.add(doc)

    db.to_disk(output_path)
    print(f"Saved {len(data)} examples to {output_path}")


convert_to_spacy(train_data, "data/spacy_training_data.spacy")
convert_to_spacy(test_data, "data/spacy_test_data.spacy")

def load_data_from_spacy_format(file_path, nlp=None):
    """
    Load training/evaluation data from a .spacy binary file.

    Parameters:
    -----------
    file_path : str
        Path to the .spacy file
    nlp : spacy.Language, optional
        spaCy language model, will create a blank one if not provided

    Returns:
    --------
    list
        List of (text, annotations) tuples in the format needed for training
    """

    # Create blank model if none provided
    if nlp is None:
        nlp = spacy.blank("id")  # Use appropriate language code

    # Load the DocBin
    doc_bin = DocBin().from_disk(file_path)

    # Convert to docs
    docs = list(doc_bin.get_docs(nlp.vocab))

    # Convert back to the training data format
    training_data = []
    for doc in docs:
        text = doc.text
        entities = []
        for ent in doc.ents:
            entities.append((ent.start_char, ent.end_char, ent.label_))

        # Create the annotation dictionary
        annotations = {"entities": entities}

        # Add to the training data list
        training_data.append((text, annotations))

    print(f"Loaded {len(training_data)} examples from {file_path}")
    return training_data

# Load data
train_data = load_data_from_spacy_format("data/spacy_training_data.spacy")
test_data = load_data_from_spacy_format("data/spacy_test_data.spacy")

nlp = create_model(train_data)

# Setup training examples
from spacy.training import Example
train_examples = []
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
train_examples.append(example)

# Training
optimizer = nlp.begin_training()
optimizer.learn_rate = 0.01

# Batch up the examples
from spacy.util import minibatch, compounding
n_iter = 30
print("Training model...")
for i in range(n_iter):
    random.shuffle(examples)
    losses = {}

    # Batch the examples
    batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, drop=0.4, losses=losses)

    print(f"Iteration {i+1}, Losses: {losses}")

# 6. Evaluate the model
def evaluate_model(nlp, eval_data):
    scorer = spacy.scorer.Scorer()
    examples = []

    for text, annotations in eval_data:
        doc_gold_text = nlp.make_doc(text)
        gold = Example.from_dict(doc_gold_text, annotations)
        pred_value = nlp(text)
        examples.append(Example(pred_value, gold.reference))

    scores = scorer.score(examples)

    # Print results
    print("Evaluation results:")
    for metric, value in scores.items():
        if metric.startswith("ents"):
            # Handle different types of values
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

evaluate_model(nlp, test_data)

val = pl.read_parquet('data/merchant_ner_labeled_validation.parquet')
validation_data = []
for row in val.iter_rows():
    validation_data.append((row[1], {"entities": [(row[5][0], row[5][1], "LOC")]}))

evaluate_model(nlp, validation_data)

# Save to disk
output_dir = "model/indonesian_location_ner_model"
nlp.to_disk(output_dir)
