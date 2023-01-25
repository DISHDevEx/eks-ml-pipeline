from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import col, count, rand, row_number
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import random



def rec_type_list_generator(input_data_type, input_split_ratio, input_rec_type_df, aggregation_column,
                            input_rec_type_features_df):
    """
    inputs
    ------
            input_data_type: String
            builds n_samples based on input string

            input_split_ratio: list
            list of split parameters

            input_rec_type_df: df
            preprocessing and filtered node df

            input_rec_type_features_df: df
            processed node features df

    outputs
    -------
            node_list: list
            randomly selected list of node id's with replacement

            input_rec_type_df: df
            final node df with newly added columns (pyspark df)

    """
    model_parameters = input_rec_type_features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]
    batch_size = model_parameters["batch_size"]

    if input_data_type == 'train':
        n_samples = batch_size * model_parameters["train_sample_multiplier"]
    elif input_data_type == 'test':
        n_samples = round(
            (batch_size * model_parameters["train_sample_multiplier"] * input_split_ratio[1]) / input_split_ratio[0])

    input_rec_type_df_freq = input_rec_type_df.groupby(aggregation_column).count().withColumnRenamed("count", "freq")
    input_rec_type_df = input_rec_type_df.join(input_rec_type_df_freq, on=aggregation_column, how='inner')
    input_rec_type_df = input_rec_type_df.filter(col('freq') > time_steps)

    print('creating the final list')
    rec_type_list = input_rec_type_df.select(aggregation_column).sample(withReplacement=True, fraction=0.85).limit(
        n_samples).rdd.flatMap(lambda x: x).collect()

    return rec_type_list, input_rec_type_df


def rec_type_ad_feature_engineering(rec_type_list_element, input_df, aggregation_column, input_features, input_scaled_features, input_time_steps, spark):
    """
     inputs
     ------
             rec_type_list_element: String
             randomly pick instance id

             input_df: df
             preprocessing and filtered node df

             input_features: list
             list of selected features

             input_scaled_features: list
             list of tobe scaled features

             input_time_steps: int
             model parameter time steps

     outputs
     -------
             rec_type_fe_df: df
             training data df for exposing it as data product (pyspark df)

     """

    spark.sparkContext.setLocalProperty("spark.scheduler.pool", rec_type_list_element)
    print('starting the loop')
    # pick random df, and normalize
    rec_type_fe_df = input_df[(input_df[aggregation_column] == rec_type_list_element)].select('*').sort("Timestamp").na.drop(
        subset=input_features)
    rec_type_fe_df_len = rec_type_fe_df.count()

    print('scaler transformation')
    # scaler transformations
    assembler = VectorAssembler(inputCols=input_features, outputCol="vectorized_features")
    scaler = StandardScaler(inputCol="vectorized_features", outputCol="scaled_features", withMean=True, withStd=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    rec_type_fe_df = pipeline.fit(rec_type_fe_df).transform(rec_type_fe_df)

    print('tensor builder')
    # tensor builder
    start = random.choice(range(rec_type_fe_df_len - input_time_steps))
    rec_type_fe_df = rec_type_fe_df.withColumn('rn', row_number().over(Window.orderBy('Timestamp'))).filter(
        (col("rn") >= start) & (col("rn") < start + input_time_steps)).drop('rn').select("*")

    spark.sparkContext.setLocalProperty("spark.scheduler.pool", None)

    return rec_type_fe_df
