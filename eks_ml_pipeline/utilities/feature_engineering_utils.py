import concurrent.futures
from functools import partial, reduce
from pyspark.sql import SparkSession


def run_multithreading(function_to_run, input_df, aggregation_column, input_features, input_scaled_features,
                       input_time_steps, selected_rec_type_list, spark):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(partial(function_to_run,
                                           input_df=input_df,
                                           aggregation_column=aggregation_column,
                                           input_features=input_features,
                                           input_scaled_features=input_scaled_features,
                                           input_time_steps=input_time_steps, spark=spark), rec_type_list_element) for
                   rec_type_list_element in selected_rec_type_list]
        df_list = [f.result() for f in futures]
    return df_list


def unionAll(*dfs):
    first, *_ = dfs
    return first.sql_ctx.createDataFrame(
        first.sql_ctx._sc.union([df.rdd for df in dfs]),
        first.schema
    )
