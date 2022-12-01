import numpy as np
import random
from utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col, count, rand
from sklearn.preprocessing import StandardScaler


def container_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):

    pyspark_container_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, filter_column_value ='Container')
    err, pyspark_container_df = pyspark_container_data.read()

    if err == 'PASS':
        #get features
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
    
        #filter inital node df based on request features
        container_df = pyspark_container_df.select("Timestamp", "kubernetes", *processed_features)
        container_df = container_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        
        # Drop NA
        cleaned_container_df = container_df.na.drop(subset=processed_features)
        
        cleaned_container_df.show(truncate=False)

        
        #Quality(timestamp filtered) nodes
        quality_filtered_container_df = cleaned_container_df.groupBy("InstanceId").agg(count("Timestamp").alias("timestamp_count"))
        quality_filtered_containers = quality_filtered_container_df.filter(col("timestamp_count").between(45,75))
        

        #Processed Node DF                                                      
        processed_container_df = cleaned_container_df.filter(col("InstanceId").isin(quality_filtered_nodes["InstanceId"]))
        
        #Null report
        null_report_df = null_report.report_generator(processed_container_df, processed_features)     
        #null_report_df.show(truncate=False)
        
        return features_df, processed_container_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    
    

def container_autoencoder_ad_feature_engineering(input_container_features_df, input_container_processed_df):
    
    model_parameters = input_container_features_df["model_parameters"]
    features =  feature_processor.cleanup(input_container_features_df["feature_name"].to_list())
    
    time_steps = model_parameters[0]["time_steps"]
    batch_size = model_parameters[0]["batch_size"]
    n_samples = batch_size * model_parameters[0]["sample_multiplier"]
    
    container_data = np.zeros((n_samples,time_steps,len(features)))
    for n in range(n_samples):
        ##pick random df, and normalize
        random_instance_df= input_container_processed_df.select("InstanceId").orderBy(rand()).limit(1)
        container_fe_df = input_container_processed_df[(input_container_processed_df["InstanceId"] == random_instance_df.first()["InstanceId"])][['Timestamp'] + features].select('*')
        container_fe_df = container_fe_df.sort("Timestamp")
        
        #scaler transformations
        assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")
        container_fe_df = assembler.transform(container_fe_df)
        scaler = StandardScaler(inputCol = "vectorized_features", outputCol = "scaled_features", withMean=True, withStd=True)
        container_fe_df = scaler.fit(container_fe_df).transform(container_fe_df)
        container_fe_df.show(truncate=False)
        
        #final X_train tensor
        start = random.choice(range(len(container_fe_df)-time_steps))
        container_data[n,:,:] = container_fe_df[start:start+time_steps][scaled_features]

    print(container_data)
    print(container_data.shape)
    
    return container_data

    
    
def container_autoencoder_train_test_split(input_df):
    
    container_train, container_test = input_df.randomSplit(weights=[0.8,0.2], seed=200)

    return container_train, container_test