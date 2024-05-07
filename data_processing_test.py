from data_ingest import IngestData
from data_processing import (find_constant_columns, delete_constant_columns, find_columns_with_fewer_values, find_duplicate_rows, find_nonnumber_columns,
                             drop_and_fill)
from feature_engineering import (bin_to_num, cat_to_col, one_hot_encoding)

# Create an instance of IngestData
ingest_data = IngestData()

# Load the data
df = ingest_data.get_data("cancer_reg.csv", encoding='latin1')

# Find constant columns
constant_columns = find_constant_columns(df)
#print("Columns that consist of single value: ", constant_columns)

fewer_columns = find_columns_with_fewer_values(df, 10)
#print("The columns with fewer values are : ", fewer_columns)

duplicated = find_duplicate_rows(df)
#print("The duplicated rows are: ", duplicated)

nonnumber = find_nonnumber_columns(df)
#print("The non number columns are: ", nonnumber)

df = bin_to_num(df)
#print(df.head())

df = cat_to_col(df)
df = one_hot_encoding(df)
df = drop_and_fill(df)
print(df.shape)

df.to_csv("cancer_processed.csv", index=False)