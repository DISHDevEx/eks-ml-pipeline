
def node_autoencoder_fe_input():
    """
    outputs
    -------
            list of parameters for node rec type
            required for autoencoder feature engineering 
            pipeline
            
    """
    ##feature parameters
    feature_group_name = "node_autoencoder_ad"
    feature_version = "v0.0.2"
    
    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "29"
    partition_hour = -1
    spark_config_setup = "384gb"
    
    ##s3 bucket parameters
    bucket = "dish-5g.core.pd.g.dp.eks.logs.e"

    
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket]


def node_pca_fe_input():
    """
    outputs
    -------
            list of parameters for node rec type
            required for pca feature engineering 
            pipeline
            
    """
    ##feature parameters
    feature_group_name = "node_pca_ad"
    feature_version = "v0.0.1"
    
    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "29"
    partition_hour = "1"
    spark_config_setup = "384gb"
    
    ##s3 bucket parameters
    bucket = "dish-5g.core.pd.g.dp.eks.logs.e"

    
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket]


def node_hmm_fe_input():
    """
    outputs
    -------
            list of parameters for node rec type
            required for hmm feature engineering 
            pipeline
            
    """
    ##feature parameters
    feature_group_name = "node_hmm_ad"
    feature_version = "v0.0.1"
    
    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "29"
    partition_hour = -1
    spark_config_setup = "384gb"
    
    ##s3 bucket parameters
    bucket = "dish-5g.core.pd.g.dp.eks.logs.e"

    
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket]


def pod_autoencoder_fe_input():
    """
    outputs
    -------
            list of parameters for pod rec type
            required for autoencoder feature engineering 
            pipeline
            
    """
    ##feature parameters
    feature_group_name = "pod_autoencoder_ad"
    feature_version = "v0.0.2"
    
    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "9"
    partition_hour = -1
    spark_config_setup = "384gb"
    
    ##s3 bucket parameters
    bucket = "dish-5g.core.pd.g.dp.eks.logs.e"

    
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket]


def pod_pca_fe_input():
    """
    outputs
    -------
            list of parameters for pod rec type
            required for pca feature engineering 
            pipeline
            
    """
    ##feature parameters
    feature_group_name = "pod_pca_ad"
    feature_version = "v0.0.1"
    
    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "9"
    partition_hour = -1
    spark_config_setup = "384gb"
    
    ##s3 bucket parameters
    bucket = "dish-5g.core.pd.g.dp.eks.logs.e"

    
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket]


def container_autoencoder_fe_input():
    """
    outputs
    -------
            list of parameters for container rec type
            required for autoencoder feature engineering 
            pipeline
            
    """
    ##feature parameters
    feature_group_name = "container_autoencoder_ad"
    feature_version = "v0.0.1"
    
    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "5"
    partition_day = "5"
    partition_hour = -1
    spark_config_setup = "384gb"
    
    ##s3 bucket parameters
    bucket = "dish-5g.core.pd.g.dp.eks.logs.e"

    
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket]


def container_pca_fe_input():
    """
    outputs
    -------
            list of parameters for container rec type
            required for pca feature engineering 
            pipeline
            
    """
    ##feature parameters
    feature_group_name = "container_pca_ad"
    feature_version = "v0.0.1"
    
    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "5"
    partition_day = "5"
    partition_hour = -1
    spark_config_setup = "384gb"
    
    ##s3 bucket parameters
    bucket = "dish-5g.core.pd.g.dp.eks.logs.e"

    
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket]

