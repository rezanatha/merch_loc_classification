import polars as pl

data = pl.read_csv('data/merchant_locs_offline_online_train.csv', separator=',', quote_char='"')


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

from sklearn.model_selection import train_test_split
X = data[['digit_ratio', 'num_blocks', 'known_offline_merchant', 'known_online_merchant', 'num_non_alphanumerics']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

import xgboost as xgb

params = {
    'objective':'binary:logistic',
    'n_estimators': 5000,
    'learning_rate':0.5,
    'max_depth':9,
    'min_child_weight':2,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'tree_method':'hist',
    'device':'cuda',
    'random_state' : 1,
    'early_stopping_rounds' :50
}

model = xgb.XGBClassifier(**params)

# Train model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True,
)

# 6. Evaluate model
from sklearn.metrics import classification_report, accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# evaluate with specific dataset
print("========evaluate with specific dataset")

data_test_2_raw = pl.read_csv('data/merchant_locs_offline_online_validation.csv', separator=',', quote_char='"').sample(100000)
data_test_2_raw = data_test_2_raw.filter(pl.col('merchant_location').is_null() & (pl.col('online_offline_flag') == 'ONLINE'))

data_test_2 = data_test_2_raw.with_columns(
    pl.col("merchant_name").map_elements(lambda x: count_digit(x)/len(x), return_dtype = pl.Float32).alias("digit_ratio"),
    pl.col("merchant_name").map_elements(lambda x: len(re.findall(r'\d+',x)), return_dtype = pl.Int32).alias("num_blocks"),
    #pl.when(pl.col("merchant_location").is_null()).then(pl.lit(0)).otherwise(pl.lit(1)).alias("location_is_known"),
    pl.when(pl.col("merchant_name").str.contains(r"((\bAFM\b)|(ALFAMART))|((\bIDM\b|INDOMARET))|(FAMILY\s?MART)|(HHB\s[A-Z0-9]{3})|(^7-(11|ELEVEN))")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_offline_merchant"),
    pl.when(pl.col("merchant_name").str.contains(r"\.CO(\.|M)|(CO\.ID)|(\.(SG|JP))|(GOOGL)|(FACEB(OO)?K)|(BILL P(A)?YM(ENT)?)")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_online_merchant"),
    pl.col("merchant_name").map_elements(lambda x: len(re.sub(r'[a-zA-Z0-9\s]','',x)), return_dtype = pl.Int32).alias("num_non_alphanumerics"),
)

data_test_2 = data_test_2.with_columns(
    pl.col("online_offline_flag").replace_strict(mapping).alias("label")
)

X_test_2 = data_test_2[['digit_ratio', 'num_blocks', 'known_offline_merchant', 'known_online_merchant', 'num_non_alphanumerics']]
y_test_2 = data_test_2['label']

y_pred = model.predict(X_test_2)
print("Accuracy:", accuracy_score(y_test_2, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_2, y_pred))

data_test_2 = data_test_2.with_columns(
    pl.lit(y_pred).alias('predicted_label')
)

data_test_2[['merchant_name', 'label', 'predicted_label']].filter(pl.col('predicted_label') != pl.col('label')).write_csv('wrong_prediction.csv')
