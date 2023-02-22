import os
from dotenv import load_dotenv
from eks_ml_pipeline import FeatureEngineeringPipeline
from eks_ml_pipeline import node_autoencoder_fe_input
import boto3


def get_parameters(vars, env):

    ##
    ## Locally the values of the env variables are the actual values, but in the
    ## real environments, the values of the env variables will be the key into SSM
    ## parameter store
    ##
    resp = {}
    if env == "local":
        for V in vars:
            resp[V] = os.getenv(V, None)
            # logger.debug(f"env var: {V} with value {resp[V]}")
    else:
        # params = list(vars.values())
        params = []
        varmap = {}
        for V in vars:
            val = os.getenv(V, None)
            params.append(val)
            varmap[V] = val

        response = ssm.get_parameters(Names=params, WithDecryption=True)
        for parameter in response["Parameters"]:
            val = parameter["Value"]
            name = parameter["Name"]
            # logger.debug("vars.items: %s" % varmap.items())
            # logger.debug("val: %s name: %s" % (val, name))
            var = [k for (k, v) in varmap.items() if v == parameter["Name"]]
            if len(var) == 0:
                continue
            var = var[0]
            resp[var] = parameter["Value"]

    return resp

if __name__ == "__main__":
    load_dotenv()
    ssm = boto3.client("ssm")

    resp = {}
    params = []
    varmap = {}
    for V in vars:
        val = os.getenv(V, None)
        params.append(val)
        varmap[V] = val

    response = ssm.get_parameters(Names=params, WithDecryption=True)
    for parameter in response["Parameters"]:
        val = parameter["Value"]
        name = parameter["Name"]
        # logger.debug("vars.items: %s" % varmap.items())
        # logger.debug("val: %s name: %s" % (val, name))
        var = [k for (k, v) in varmap.items() if v == parameter["Name"]]
        if len(var) == 0:
            continue
        var = var[0]
        resp[var] = parameter["Value"]

    rec_type = 'Node'
    compute_type = 'sagemaker'
    aggregation_column = 'InstanceId'
    input_data_type = 'test'

    fep = FeatureEngineeringPipeline(node_autoencoder_fe_input(), rec_type, compute_type, input_data_type)

    fep.run_preproceesing()