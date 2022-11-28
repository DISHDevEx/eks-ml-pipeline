from pyspark.sql.functions import col, count, when, isnan


def report_generator(input_df, features_list):
    
    null_report_df = input_df.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in features_list])
    
    return null_report_df