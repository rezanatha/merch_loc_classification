import polars as pl
import re
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load validation data
predict_data = pl.read_csv('data/merchant_locs_offline_online_data_example.txt', separator=',', quote_char='"').sample(100000)

# Extract features
feature_cols = ['digit_ratio', 'num_blocks', 'known_offline_merchant', 'known_online_merchant', 'num_non_alphanumerics']
predict_data = predict_data.with_columns(
    pl.col("merchant_name").map_elements(lambda x: count_digit(x)/len(x), return_dtype=pl.Float32).alias("digit_ratio"),
    pl.col("merchant_name").map_elements(lambda x: len(re.findall(r'\d+',x)), return_dtype=pl.Int32).alias("num_blocks"),
    pl.when(pl.col("merchant_name").str.contains(r"((\bAFM\b)|(ALFAMART))|((\bIDM\b|INDOMARET))|(FAMILY\s?MART)|(HHB\s[A-Z0-9]{3})|(^7-(11|ELEVEN))")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_offline_merchant"),
    pl.when(pl.col("merchant_name").str.contains(r"\.CO(\.|M)|(CO\.ID)|(\.(SG|JP))|(GOOGL)|(FACEB(OO)?K)|(BILL P(A)?YM(ENT)?)")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("known_online_merchant"),
    pl.col("merchant_name").map_elements(lambda x: len(re.sub(r'[a-zA-Z0-9\s]','',x)), return_dtype=pl.Int32).alias("num_non_alphanumerics"),
)

predict_data = predict_data[feature_cols]
# Evaluate on validation set
y = model.predict(predict_data)

result = predict_data.with_columns(
    pl.lit(y).alias('predicted_label')
)
