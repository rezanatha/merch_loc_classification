import polars as pl
import re
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Start MLflow run
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.start_run(run_name="merchant_classification")


# Load data
data = pl.read_csv('data/merchant_locs_offline_online_train.csv', separator=',', quote_char='"')

# Log dataset info
mlflow.log_param("train_data_path", 'data/merchant_locs_offline_online_train.csv')
mlflow.log_param("dataset_size", len(data))

# Prepare target variable
mapping = {"ONLINE": 1, "OFFLINE": 0}
data = data.with_columns(
    pl.col("online_offline_flag").replace_strict(mapping).alias("label")
)

# Feature extraction function
def count_digit(x):
    return sum(c.isdigit() for c in x)

# Log feature extraction parameters
mlflow.log_param("digit_ratio_feature", "count_digit(x)/len(x)")
mlflow.log_param("num_blocks_feature", "count of digit blocks using regex")
mlflow.log_param("known_offline_merchant_feature", r"((\bAFM\b)|(ALFAMART))|((\bIDM\b|INDOMARET))|(FAMILY\s?MART)|(HHB\s[A-Z0-9]{3})|(^7-(11|ELEVEN))")
mlflow.log_param("known_online_merchant_feature", r"\.CO(\.|M)|(CO\.ID)|(\.(SG|JP))|(GOOGL)|(FACEB(OO)?K)|(BILL P(A)?YM(ENT)?)")
mlflow.log_param("num_non_alphanumerics_feature", "count of non-alphanumeric characters")

# Feature extraction
data = data.with_columns(
    pl.col("merchant_name").map_elements(lambda x: count_digit(x)/len(x), return_dtype=pl.Float32).alias("digit_ratio"),
    pl.col("merchant_name").map_elements(lambda x: len(re.findall(r'\d+',x)), return_dtype=pl.Int32).alias("num_blocks"),
    pl.when(pl.col("merchant_name").str.contains(r"((\bAFM\b)|(ALFAMART))|((\bIDM\b|INDOMARET))|(FAMILY\s?MART)|(HHB\s[A-Z0-9]{3})|(^7-(11|ELEVEN))")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_offline_merchant"),
    pl.when(pl.col("merchant_name").str.contains(r"\.CO(\.|M)|(CO\.ID)|(\.(SG|JP))|(GOOGL)|(FACEB(OO)?K)|(BILL P(A)?YM(ENT)?)")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_online_merchant"),
    pl.col("merchant_name").map_elements(lambda x: len(re.sub(r'[a-zA-Z0-9\s]','',x)), return_dtype=pl.Int32).alias("num_non_alphanumerics"),
)

# Log feature statistics
feature_cols = ['digit_ratio', 'num_blocks', 'known_offline_merchant', 'known_online_merchant', 'num_non_alphanumerics']

# Split data
X = data[feature_cols]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
mlflow.log_param("test_size", 0.2)
mlflow.log_param("random_state", 1)
mlflow.log_param("train_size", len(X_train))
mlflow.log_param("test_size_actual", len(X_test))

# Define hyperparameters
params = {
    'objective': 'binary:logistic',
    'n_estimators': 5000,
    'learning_rate': 0.5,
    'max_depth': 9,
    'min_child_weight': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 1,
    'early_stopping_rounds': 50
}

# Log all hyperparameters
for param_name, param_value in params.items():
    mlflow.log_param(param_name, param_value)

# Initialize model
model = xgb.XGBClassifier(**params)

# Train model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True,
)

# Log best iteration
mlflow.log_param("best_iteration", model.best_iteration)

# Evaluate model on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Log metrics
mlflow.log_metric("test_accuracy", accuracy)
mlflow.log_metric("test_precision", precision)
mlflow.log_metric("test_recall", recall)
mlflow.log_metric("test_f1", f1)
mlflow.log_metric("test_roc_auc", roc_auc)

# Print evaluation results
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Log feature importance
feature_importance = model.feature_importances_
for i, col in enumerate(feature_cols):
    mlflow.log_metric(f"importance_{col}", feature_importance[i])

# Log model
mlflow.xgboost.log_model(model, "model")

# Evaluate with specific dataset
print("========evaluate with specific dataset")

# Load validation data
data_test_2_raw = pl.read_csv('data/merchant_locs_offline_online_validation.csv', separator=',', quote_char='"').sample(100000)
mlflow.log_param("validation_data_path", 'data/merchant_locs_offline_online_validation.csv')
mlflow.log_param("validation_sample_size", 100000)

# Filter validation data
data_test_2_raw = data_test_2_raw.filter(pl.col('merchant_location').is_null() & (pl.col('online_offline_flag') == 'ONLINE'))
mlflow.log_param("validation_filtered_size", len(data_test_2_raw))

# Apply feature extraction to validation data
data_test_2 = data_test_2_raw.with_columns(
    pl.col("merchant_name").map_elements(lambda x: count_digit(x)/len(x), return_dtype=pl.Float32).alias("digit_ratio"),
    pl.col("merchant_name").map_elements(lambda x: len(re.findall(r'\d+',x)), return_dtype=pl.Int32).alias("num_blocks"),
    pl.when(pl.col("merchant_name").str.contains(r"((\bAFM\b)|(ALFAMART))|((\bIDM\b|INDOMARET))|(FAMILY\s?MART)|(HHB\s[A-Z0-9]{3})|(^7-(11|ELEVEN))")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_offline_merchant"),
    pl.when(pl.col("merchant_name").str.contains(r"\.CO(\.|M)|(CO\.ID)|(\.(SG|JP))|(GOOGL)|(FACEB(OO)?K)|(BILL P(A)?YM(ENT)?)")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_online_merchant"),
    pl.col("merchant_name").map_elements(lambda x: len(re.sub(r'[a-zA-Z0-9\s]','',x)), return_dtype=pl.Int32).alias("num_non_alphanumerics"),
)

data_test_2 = data_test_2.with_columns(
    pl.col("online_offline_flag").replace_strict(mapping).alias("label")
)

X_test_2 = data_test_2[feature_cols]
y_test_2 = data_test_2['label']

# Evaluate on validation set
y_pred_val = model.predict(X_test_2)
y_pred_proba_val = model.predict_proba(X_test_2)[:, 1]

# Calculate validation metrics
val_accuracy = accuracy_score(y_test_2, y_pred_val)
val_precision = precision_score(y_test_2, y_pred_val)
val_recall = recall_score(y_test_2, y_pred_val)
val_f1 = f1_score(y_test_2, y_pred_val)
val_roc_auc = roc_auc_score(y_test_2, y_pred_proba_val)

# Log validation metrics
mlflow.log_metric("validation_accuracy", val_accuracy)
mlflow.log_metric("validation_precision", val_precision)
mlflow.log_metric("validation_recall", val_recall)
mlflow.log_metric("validation_f1", val_f1)
mlflow.log_metric("validation_roc_auc", val_roc_auc)

# Print validation results
print("Validation Accuracy:", val_accuracy)
print("\nValidation Classification Report:")
print(classification_report(y_test_2, y_pred_val))

# Save wrong predictions
data_test_2 = data_test_2.with_columns(
    pl.lit(y_pred_val).alias('predicted_label')
)

wrong_predictions = data_test_2[['merchant_name', 'label', 'predicted_label']].filter(pl.col('predicted_label') != pl.col('label'))
wrong_predictions.write_csv('wrong_prediction.csv')
mlflow.log_artifact('wrong_prediction.csv')

# Log wrong prediction count
mlflow.log_metric("wrong_prediction_count", len(wrong_predictions))

# End MLflow run
mlflow.end_run()
