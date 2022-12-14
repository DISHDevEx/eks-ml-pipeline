from msspackages import Pyspark_data_ingestion
import pandas as pd


def inference_data_builder(input_year, input_month,  input_day, input_hour, rec_type, input_setup):

    if input_hour == -1:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}_{input_day}'
    elif input_day == -1:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}'
    else:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}_{input_day}_{input_hour}'
    
    pyspark_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value = rec_type)
    err, pyspark_df = pyspark_data.read()
    
    #if err == 'PASS':
    print(err)
    pyspark_df = pyspark_df.select("*")
    pyspark_df.write.mode('overwrite').parquet(f's3a://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/{file_name}/')


if __name__ == "__main__":
    inference_data_builder()