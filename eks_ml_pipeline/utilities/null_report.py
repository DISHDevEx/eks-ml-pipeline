from pyspark.sql.functions import col, count, when, isnan


def report_generator(input_df, features_list):
    """
    Input:
    -----
    input_df: df
    df to run the null report
    
    
    features_list: list
    list of columns to loop through
    
    Output:
    ------
    null_report_df: df
    df report of all the columns in the input df
    """
    
    null_report_df = input_df.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in features_list])
    
    return null_report_df