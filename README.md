Data preprocessing, feature engineering, training and deployment of eks pattern detection models.


### __Project Structure__

Initial project structure for the eks-ml-pipeline. This will evolve over the time to reflect latest changes. 

```

├──  deployment_notebooks				contains notebooks consisting of deployment pipeline and inferencing
│   ├── node_autoencoder_ad_v0.0.2_2022_9_29.ipynb
│
│
├──  feature_engineering				contains all feature engineering modules by model type
│   ├── container_autoencoder_pca_ad.py
│   ├── node_autoencoder_pca_ad.py
│   ├── node_hmm_ad.py
│   ├── pod_autoencoder_pca_ad.py
│   ├── train_test_split.py
│
│
├──  inputs						contains functions for input parameters for feature engineering, training, and inferencing pipelines
│   ├── feature_engineering_input.py
│   ├── inference_input.py
│   ├── training_input.py
│
│
├──  models						contains all the modeling classes to initialize, fit and test models
│   ├── autoencoder_model.py
│   ├── pca_model.py
│
│
└── tests						contains unit and integration tests of the project
│    ├── unit
│    └── integration
│
│
└── utilities					any additional utilties we need for the project
│    └── feature_processor.py  
│    └── null_report.py
│    └── s3_utilities.py 
│    └── variance_loss.py 
│ 

```


### __Running EMR Serverless jobs__

__us-east-1 applications:__
* pd-autoencoder-ad-v1 : **00f64bef5869kl09**
* pd-autoencoder-ad-v2 : **00f66ohicnjchu09**
* pd-test-s3-writes : **00f66mmuts7enm09**

__us-west-2 applications:__ 
* pd-autoencoder-ad-container-v1  : **00f672mqiak1fp0l**

**Note**: while launching your job, please make note of the region from where you are running it.
jobs for us-east-1 applications can only be launched from us-east-1 and similarly, jobs for us-west-2 applications can only be launched from us-west-2. 
#### __Usage__
##### __Scenario 1: From CLI__

Run the following command:
```console
python emr_serverless.py --job-role-arn <<job_role_arn>> --applicationId <<applicationID>> --s3-bucket <<s3_bucket_name>> --entry-point <<emr_entry_point>> --zipped-env <<zipped_env_path>> --custom-spark-config <<custom_spark_config>>
```
Optional arguments:
- __--job-role-arn__    : default value = 'arn:aws:iam::064047601590:role/Pattern-Detection-EMR-Serverless-Role'
- __--custom-spark-config__   : default value = default
    
Without optional arguments:
```console
python emr_serverless.py --applicationId <<applicationID>> --s3-bucket <<s3_bucket_name>> --entry-point <<emr_entry_point>> --zipped-env <<zipped_env_path>>
```

For examples on how to run the jobs via CLI, refer to the documentation [here](https://dish-wireless-network.atlassian.net/wiki/spaces/MSS/pages/327549297/EMR+Serverless+-+How+To+Guide#Scenario-1%3A-From-CLI).


##### __Scenario 2: From Sagemaker Notebook__

The notebook should be in the ```'/root/eks-ml-pipeline'``` path.
Follow the below steps to configure the basic setup to launch EMR Serverless apllications from Sagemaker Notebook:

1. Check the path 
```console
pwd
```
2. If not already installed, install msspackages by using the .whl file (this assumes that the whl file already exists in the below location)
```console
!pip install /root/msspackages/dist/msspackages-0.0.7-py3-none-any.whl
```
3. Install the necessary requirements
```console
pip install -r requirements.txt
```
4. Import the EMRServerless class
```console
from eks_ml_pipeline import EMRServerless
```
For detailed steps on how to submit a new job to EMR serverless application, refer to the documentaion [here](https://dish-wireless-network.atlassian.net/wiki/spaces/MSS/pages/327549297/EMR+Serverless+-+How+To+Guide#2.a-When-submitting-a-new-job-to-EMR-serverless-application).
