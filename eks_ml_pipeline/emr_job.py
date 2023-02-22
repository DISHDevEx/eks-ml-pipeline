from eks_ml_pipeline import FeatureEngineeringPipeline
from eks_ml_pipeline import node_autoencoder_fe_input

if __name__ == "__main__":
    rec_type = 'Node'
    compute_type = 'emr'
    input_data_type = 'test'

    print('Initialize fe pipeline')
    fep = FeatureEngineeringPipeline(node_autoencoder_fe_input(), rec_type, compute_type, input_data_type)

    print('Run pre-processing')
    fep.run_preproceesing()
