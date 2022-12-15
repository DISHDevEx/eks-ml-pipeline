from msspackages import Pyspark_data_ingestion
import pandas as pd
import awswrangler as wr
    


def inference_data_builder(input_year, input_month,  input_day, input_hour, rec_type, input_setup):
    
    """
    inputs
    ------
            input_year : STRING | Int
            the year from which to read data, leave empty for all years

            input_month : STRING | Int
            the month from which to read data, leave empty for all months

            input_day : STRING | Int
            the day from which to read data, leave empty for all days

            input_hour: STRING | Int
            the hour from which to read data, leave empty for all hours
            
            rec_type: STRING
            uses schema for rec type when building pyspark df
            
            input_setup: STRING 
            kernel config
    
    outputs
    -------
            writes parquet to specific s3 path
    """

    if input_hour == -1:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}_{input_day}'
    elif input_day == -1:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}'
    else:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}_{input_day}_{input_hour}'
    
    pyspark_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value = rec_type)
    err, pyspark_df = pyspark_data.read()
    
    if err == 'PASS':
        print(err)
        pyspark_df = pyspark_df.toPandas()
        wr.s3.to_parquet(pyspark_df, path=f"s3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/{file_name}.parquet")


if __name__ == "__main__":
    inference_data_builder()