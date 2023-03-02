from eks_ml_pipeline import FeatureEngineeringPipeline, container_autoencoder_fe_input
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("EMRServerless").getOrCreate()

    rec_type = 'Container'
    compute_type = 'emr'
    input_data_type = 'train'

    print('Initialize fe pipeline')
    fep = FeatureEngineeringPipeline(container_autoencoder_fe_input(), rec_type, compute_type, input_data_type)

    print('Run pre-processing')
    fep.run_preprocessing()