""" Utilities file to perform common feature engineering tasks for each rec type """
import concurrent.futures
from functools import partial, reduce
from pyspark.sql import SparkSession


def run_multithreading(function_to_run, input_df, aggregation_column, input_features, input_scaled_features,
                       input_time_steps, selected_rec_type_list, spark):
    """
        inputs
        ------
                function_to_run: func
                function to run using multithreading

                input_df: df
                input df to perform feature engineering

                aggregation_column: String
                column to perform aggregation on

                input_features: list
                input feature list

                input_scaled_features: list
                scaled features list

                input_time_steps: String
                time steps

                selected_rec_type_list: list
                randomly selected list of record type id's

                spark:
                spark session

        outputs
        -------
                df_list: list
                list of dataframes

                tensor_list: list
                final list of tensors
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(partial(function_to_run,
                                           input_df=input_df,
                                           aggregation_column=aggregation_column,
                                           input_features=input_features,
                                           input_scaled_features=input_scaled_features,
                                           input_time_steps=input_time_steps, spark=spark), rec_type_list_element) for
                   rec_type_list_element in selected_rec_type_list]
        df_list = [f.result()[0] for f in futures]
        tensor_list = [f.result()[1] for f in futures]
    return df_list, tensor_list


def unionAll(*dfs):
    """
        inputs
        ------
                dfs: list
                list of dfs to join

        outputs
        -------
                combined dataframe
    """
    first, *_ = dfs
    return first.sql_ctx.createDataFrame(
        first.sql_ctx._sc.union([df.rdd for df in dfs]),
        first.schema
    )
