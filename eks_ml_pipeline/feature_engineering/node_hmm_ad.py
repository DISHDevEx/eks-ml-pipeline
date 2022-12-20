
import numpy as np
import random
from ..utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col, count, rand, get_json_object
from pyspark.ml.feature import  StandardScaler
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.ml import Pipeline

def node_hmm_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):
    """
    inputs
    ------
            
            feature_group_name: String
            json name to get the required features
            
            feature_group_created_date: STRING
            json created date to get the latest features 
            
            input_year : STRING | Int
            the year from which to read data, leave empty for all years

            input_month : STRING | Int
            the month from which to read data, leave empty for all months

            input_day : STRING | Int
            the day from which to read data, leave empty for all days

            input_hour: STRING | Int
            the hour from which to read data, leave empty for all hours
            
    
    outputs
    -------
            
            processed_node_df: preprocessed dataframe
            
    """
    
    node_data = Pyspark_data_ingestion(
        year = input_year,
        month = input_month, 
        day = input_day, 
        hour = input_hour, 
        filter_column_value ='Node',
        setup='128gb')
    spark = node_data.get_spark()
    err, node_df = node_data.read()
    
    if err == 'PASS':
        
        #get features
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
    
        #filter inital node df based on request features
        node_df = pyspark_node_df.select("Timestamp", "Instance_id", *processed_features)
        
        node_df = node_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("integer"))
        
        
        # Drop NA
        cleaned_node_df = node_df.na.drop(subset=processed_features)

        
        #Quality(timestamp filtered) nodes
        quality_filtered_node_df = cleaned_node_df.groupBy("Instance_id").agg(count("Timestamp").alias("timestamp_count"))
        quality_filtered_nodes = quality_filtered_node_df.filter(col("timestamp_count").between(45*hour_params,75*hour_params))
        

        #Processed Node DF                                                      
        processed_node_df = cleaned_node_df.filter(col("Instance_id").isin(quality_filtered_nodes["Instance_id"]))
        
        #Null report
        null_report_df = null_report.report_generator(processed_node_df, processed_features)     
        #null_report_df.show(truncate=False)
        
        return  features_df, processed_node_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    
    
def node_hmm_ad_feature_engineering(input_node_features_df, input_node_processed_df):
    """
    inputs
    ------
            
            input_pod_processed_df: df
            preprocessing and filtered node df 
    
    outputs
    -------
            
            final_pod_fe_df: df
            training data df for exposing it as data product
            
    """


    #scaler transformation
    # this function intends to normalize the node data by each node
    
    model_parameters = input_node_features_df["model_parameters"].iloc[0]
    features =  feature_processor.cleanup(input_node_features_df["feature_name"].to_list())
    
    time_steps = model_parameters["time_steps"]
    hour_params = model_parameters["hour_params"]
    weight = model_parameters["weight"]

    n = 0
    samplesize = hour_params*1440*weight/time_steps
    final_df = None
    while n < samplesize:


        ##pick random node
        random_instance = random.choice(input_df.select("Instance_id").rdd.flatMap(list).collect())
        node_df = input_df[(input_df["instance_id"] ==  random_instance)][["Timestamp", "Instance_id"] + features].select('*')
        node_df = node_df.sort("Timestamp")
        node_df = node_df.na.drop(subset=features)

        #fix negative number issue
        if node_df.count()-time_steps<= 0:
            print(f'Exception occurred: not enough data')
            continue
            
         #standardize data from the node
        assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")
        scaler = StandardScaler(inputCol = "vectorized_features", outputCol = "scaled_features", withMean=True, withStd=True)
        pipeline = Pipeline(stages=[assembler, scaler])
        node_df = pipeline.fit(node_df).transform(node_df)

        #pick random time slice of 12 timestamps from this node
        start = random.choice(range(node_df.count()-time_steps))
        node_slice_df = node_df.withColumn('rn', row_number().over(Window.orderBy("Timestamp"))).filter((col("rn") >= start) & (col("rn") < start+time_steps))
        node_slice_df = node_slice_df.select('Timestamp','scaled_features')

        #fill the large dataset
        if not final_df:
            final_df = node_slice_df
        else:
            final_df = final_df.union( node_slice_df)

        print(f'Finished with sample #{n}')

        n += 1

    
    #group by timestamp to take average value for the same timestamp
        final_df = final_df.sort("Timestamp")
        final_df =  final_df.groupBy("Timestamp").agg(mean("scaled_features").alias("scaled_features"))


    
    
    return final_df
    
    

    
    
def node_hmm_train_test_split(input_df,split = 0.5):
    """
    inputs
    ------
            
            input_df: df
            preprocessing and filtered node df 
            
            weight: float
            select weight of split
            
    
    outputs
    -------
            
            pod_train: df
            training data df for exposing it as data product
            
            
            pod_test: df
            testing data df for exposing it as data product
            
    """

    
    
    
    node_train, node_test = input_df.randomSplit(weights=[split,1-split], seed=200)

  
      
    
    
    return node_train, node_test
    

    