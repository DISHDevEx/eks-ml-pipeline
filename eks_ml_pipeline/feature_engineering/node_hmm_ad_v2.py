"""
Contributed by Ruyi Yang
MSS Dish 5g - Pattern Detection

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""

def node_hmm_ad_v2(feature_group_name, feature_group_version, input_year, input_month, input_day, input_hour, input_setup = "default"):
    
    """
    inputs
    ------
            feature_group_name: STRING
            json name to get the required features
            
            feature_group_version: STRING
            json version to get the latest features 
            
            input_year : STRING | Int
            the year from which to read data, leave empty for all years

            input_month : STRING | Int
            the month from which to read data, leave empty for all months

            input_day : STRING | Int
            the day from which to read data, leave empty for all days

            input_hour: STRING | Int
            the hour from which to read data, leave empty for all hours
            
            input_setup: STRING 
            kernel config
    
    outputs
    -------
            features_df : processed features dataFrame
            final_pod_df: pre processed node dataframe
            
    """

    node_data = Pyspark_data_ingestion(
        year = input_year, 
        month = input_month, 
        day = input_day, 
        hour = input_hour, 
        setup = input_setup, 
        filter_column_value ='Node')
    err, node_df = node_data.read()
    node_df = node_df.select("InstanceId",'Timestamp','node_cpu_utilization','node_memory_utilization')

 
    if err == 'PASS':
        
        #get features
        features_df = get_features(feature_group_name,feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
        
        model_parameters = features_df["model_parameters"].iloc[0]
  
        #drop na values in node cpu and memory utilization
        node_df = node_df.select("InstanceId","Timestamp", *processed_features)
        node_df = node_df.na.drop(subset=processed_features)
        
        #remove nodes which has a time gap over 2 minutes (epochtime = 2*60*1000=120000)
        w = Window.partitionBy('InstanceId').orderBy('Timestamp')
        node_df = node_df.withColumn('lead', f.lag('Timestamp', 1).over(w)) \
              .withColumn(
                'Timediff', 
                f.when(f.col('lead').isNotNull(), 
                f.col('Timestamp') - f.col('lead'))
                .otherwise(f.lit(None)))
               
        
        temp_df = node_df\
            .groupby("InstanceId")\
            .max("Timediff")\
            .select('InstanceId',f.col('max(TimeDiff)').alias('maxDiff'))\
            .filter("maxDiff<=120000")
                                                             
        node_df = node_df.filter(col("InstanceId").isin(temp_df['InstanceId']))
        node_df = node_df.sort("InstanceId","Timestamp")
        node_df = node_df.select('InstanceId','Timestamp','node_cpu_utilization','node_memory_utilization')
        
        #Drop rows with nans 
        node_df = node_df.na.drop("all")
           
        
        return node_df
    else:
        empty_df = pd.DataFrame()
        return empty_df
    
    
def node_hmm_train_test_split(input_df,split = 0.5):
    
    """
    inputs
    ------
            
            input_df: df
            preprocessing node df 
            
            weight: float
            select weight of split
            
    
    outputs
    -------
            
            pod_train: df
            training data df for exposing it as data product
            
            
            pod_test: df
            testing data df for exposing it as data product
            
    """
    temp_df = input_df.select('InstanceId')
    node_train_id, node_test_id = temp_df.randomSplit(weights=[split,1-split], seed=200)  
    node_train = input_df.filter(col("InstanceId").isin(node_train_id['InstanceId']))
    node_test = input_df.filter(col("InstanceId").isin(node_test_id['InstanceId']))
    
    return node_train, node_test

def feature_engineering(input_df):
    
    """
    inputs
    ------
            
            input_df: df
            preprocessing node df 
            
            
    outputs
    -------
            
            features_list: list
            nested list in which each element is a 2-D nparray
            
    """
    
    #sort data
    input_df = input_df.sort('InstanceId','Timestamp')
    
    #get features
    features_df = get_features(feature_group_name,feature_group_version)
    features = features_df["feature_name"].to_list()
    
    #standardize feature data from the node
    features = ['node_cpu_utilization','node_memory_utilization']
    w = Window.partitionBy('InstanceId')
    for c in features:
        input_df = (input_df.withColumn('mean', f.min(c).over(w))
            .withColumn('std', f.max(c).over(w))
            .withColumn(c, ((f.col(c) - f.col('mean')) / (f.col('std'))))
            .drop('mean')
            .drop('std'))
        
    #standard scale the data
    vecAssembler = VectorAssembler(inputCols=["node_cpu_utilization", "node_memory_utilization"], outputCol="features")
    node_train = vecAssembler.transform(node_train)
    node_train = node_train.select('InstanceId','features')
        
    #transfer data to a nested list (#timestamps * #features for each node)
    instance_list = node_train.select('InstanceId').distinct()
    features_list = []
    for instance in instance_list:
        sub = node_train.filter(node_train.InstanceId == instance)
        sub_features = np.array(sub.select("features").collect())
        features_list.append(sub_features)
                                
    return features_list