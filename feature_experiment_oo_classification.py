import polars as pl
import torch

device = 'cuda'
torch.set_default_device(device)

data = pl.read_csv('data/merchant_locs_train.csv', separator=',', quote_char='"').sample(200000)

mapping = {"ONLINE": 1, "OFFLINE": 0}
data = data.with_columns(
    pl.col("online_offline_flag").replace_strict(mapping).alias("label")
)

import re
def count_digit(x):
    return sum(c.isdigit() for c in x)

data = data.with_columns(
    pl.col("merchant_name").map_elements(lambda x: count_digit(x)/len(x), return_dtype = pl.Float32).alias("digit_ratio"),
    pl.col("merchant_name").map_elements(lambda x: len(re.findall(r'\d+',x)), return_dtype = pl.Int32).alias("num_blocks"),
    #pl.when(pl.col("merchant_location").is_null()).then(pl.lit(0)).otherwise(pl.lit(1)).alias("location_is_known"),
    pl.when(pl.col("merchant_name").str.contains(r"((\bAFM\b)|(ALFAMART))|((\bIDM\b|INDOMARET))|(FAMILY\s?MART)|(HHB\s[A-Z0-9]{3})|(^7-(11|ELEVEN))")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_offline_merchant"),
    pl.when(pl.col("merchant_name").str.contains(r"\.CO(\.|M)|(CO\.ID)|(\.(SG|JP))|(GOOGL)|(FACEB(OO)?K)|(BILL P(A)?YM(ENT)?)")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_online_merchant"),
    pl.col("merchant_name").map_elements(lambda x: len(re.sub(r'[a-zA-Z0-9\s]','',x)), return_dtype = pl.Int32).alias("num_non_alphanumerics"),
)

# bert auto tokenizer
import transformers, tqdm

bert_tokenizer = transformers.AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
bert_model = transformers.AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
bert_model.to('cuda')
bert_model.eval()

def create_bert_feature(data, batch_size=4):
    all_embeddings = []

    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = list(data[i:i + batch_size])

        # Tokenize the current batch
        bert_input = bert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)

        # # Move inputs to GPU if available
        # if torch.cuda.is_available():
        #     bert_input = {k: v.cuda() for k, v in bert_input.items()}

        # Get embeddings
        with torch.no_grad():
            bert_output = bert_model(**bert_input)

        # Extract the [CLS] token embedding for each example in the batch
        batch_embeddings = bert_output.last_hidden_state[:, 0, :]

        all_embeddings.append(batch_embeddings)
        torch.cuda.empty_cache()

    # Concatenate all batches into one tensor
    return torch.cat(all_embeddings, dim=0)

# def create_bert_feature(data):
#     bert_input = bert_tokenizer(list(data), return_tensors="pt", padding=True, truncation=True)
#     # Get embeddings
#     with torch.no_grad():
#         bert_output = bert_model(**bert_input)
#     return bert_output.last_hidden_state[:, 0, :]

##### SPECIAL DATA FOR EVALUATION (OFFLINE ONLY)

data_test_2_raw = pl.read_csv('data/merchant_locs_train.csv', separator=',', quote_char='"').sample(100000)
data_test_2_raw = data_test_2_raw.filter(pl.col('merchant_location').is_null() & (pl.col('online_offline_flag') == 'OFFLINE'))

data_test_2_raw['online_offline_flag'].value_counts()

data_test_2 = data_test_2_raw.with_columns(
    pl.col("merchant_name").map_elements(lambda x: count_digit(x)/len(x), return_dtype = pl.Float32).alias("digit_ratio"),
    pl.col("merchant_name").map_elements(lambda x: len(re.findall(r'\d+',x)), return_dtype = pl.Int32).alias("num_blocks"),
    #pl.when(pl.col("merchant_location").is_null()).then(pl.lit(0)).otherwise(pl.lit(1)).alias("location_is_known"),
    pl.when(pl.col("merchant_name").str.contains(r"((\bAFM\b)|(ALFAMART))|((\bIDM\b|INDOMARET))|(FAMILY\s?MART)|(HHB\s[A-Z0-9]{3})|(^7\-(11|ELEVEN))")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_offline_merchant"),
    pl.when(pl.col("merchant_name").str.contains(r"\.CO(\.|M)|(CO\.ID)|(\.(SG|JP))|(GOOGL)|(FACEB(OO)?K)|(BILL P(A)?YM(ENT)?)")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_online_merchant"),
    pl.col("merchant_name").map_elements(lambda x: len(re.sub(r'[a-zA-Z0-9\s]','',x)), return_dtype = pl.Int32).alias("num_non_alphanumerics"),
)


data_test_2 = data_test_2.with_columns(
    pl.col("online_offline_flag").replace_strict(mapping).alias("label")
)

## TRACK FEATURE EXPERIMENTATIONS
import mlflow
from mlflow.models import infer_signature
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

basic_features = ['digit_ratio', 'num_blocks', 'known_offline_merchant', 'known_online_merchant', 'num_non_alphanumerics']

feature_sets = {
    'basic':(basic_features, []),
    'basic_with_bert':(basic_features, ['bert'])
}

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Feature Experiment")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

def evaluate_feature_sets(X,y, feature_set_name, features):
    with mlflow.start_run(run_name=f'feature_experiment_{feature_set_name}'):
        # Log feature set metadata
        experiment_id = mlflow.active_run().info.experiment_id
        mlflow.log_param("feature_set", feature_set_name)

        has_bert = features[1] == ['bert']
        num_features = len(features[0]) + 312 if has_bert else len(features[0])
        mlflow.log_param("num_features", num_features)
        mlflow.log_param("features", features)

        if has_bert:
            bert_feature = create_bert_feature(data['merchant_name'])
            X = torch.cat([torch.tensor(X[features[0]].to_numpy()).to(device), bert_feature], dim=1)
        else:
            X = torch.tensor(X[features[0]].to_numpy()).to(device)
        y = torch.tensor(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        params = {
                'objective':'binary:logistic',
                'n_estimators': 100,
                'learning_rate':0.2,
                'max_depth':6,
                'min_child_weight':3,
                'subsample':0.8,
                'colsample_bytree':0.8,
                'tree_method':'hist',
                'device':'cuda',
                'random_state' : 1,
                'early_stopping_rounds' :20
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])


        # 6. Evaluate model

        y_pred = model.predict(X_test)
        # Log multiple metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test.to('cpu'), y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test.to('cpu'), y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test.to('cpu'), y_pred))

        # Evaluate model to specific dataset (accuracy only)
        if has_bert:
            data_test_bert = create_bert_feature(data_test_2['merchant_name'])
            X_test_specific = torch.cat([torch.tensor(data_test_2[features[0]].to_numpy()).to(device), data_test_bert],
                     dim=1)
        else:
            X_test_specific = torch.tensor(data_test_2[features[0]].to_numpy()).to(device)

        y_test_specific = torch.tensor(data_test_2['label'])
        y_pred = model.predict(X_test_specific)
        mlflow.log_metric("accuracy_specific", accuracy_score(y_test_specific.to('cpu'), y_pred))

        # Save report
        X_test_specific = data_test_2.with_columns(
            pl.lit(y_pred).alias('predicted_label')
        )
        X_test_specific[['merchant_name','known_offline_merchant','known_online_merchant','label','predicted_label']].filter(pl.col('predicted_label') != pl.col('label')).write_csv(f"prediction_result_{feature_set_name}.csv")
        mlflow.log_artifact(f"prediction_result_{feature_set_name}.csv")

        # Log model
        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(model, "model",signature=signature)


X = data.drop('label')
y = data['label']

for feature_name, features in feature_sets.items():
    evaluate_feature_sets(X,y,feature_name, features)
