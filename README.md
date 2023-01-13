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

<<<<<<< HEAD

### __Running EMR Serverless jobs from Sagemaker__
=======
### __ inital project setup__
1. Check the path
```console
!pwd
```
2. If not already installed, install msspackages by using the .whl file (this assumes that the whl file already exists in the below location)
```console
!pip install /root/msspackages/dist/msspackages-0.0.7-py3-none-any.whl
```
3. Install the necessary requirements
```console
!pip install -r requirements.txt
```
4. run below function to install java dependencies to run pyspark jobs
```console
from msspackages import setup_runner
setup_runner()
```

### __Running EMR Serverless jobs__
>>>>>>> origin

__us-east-1 applications:__
* pattern-detection-emr-serverless : **00f6muv5dgv8n509**
* pd-test-s3-writes : **00f66mmuts7enm09**

__us-west-2 applications:__ 
* pattern-detection-emr-serverless  : **00f6mv29kbd4e10l**

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

<<<<<<< HEAD
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
For detailed step on how to submit a new job to EMR serverless application, refer to the documentaion [here](https://dish-wireless-network.atlassian.net/wiki/spaces/~6329e5517f85f167779caffe/pages/318669446/EMR+Serverless+-+How+to+Guide#2.a-When-submitting-a-new-job-to-EMR-serverless-application).
=======
1. Import the EMRServerless class
```console
from eks_ml_pipeline import EMRServerless
```
For detailed steps on how to submit a new job to EMR serverless application, refer to the documentaion [here](https://dish-wireless-network.atlassian.net/wiki/spaces/MSS/pages/327549297/EMR+Serverless+-+How+To+Guide#2.a-When-submitting-a-new-job-to-EMR-serverless-application).

### __Running Feature Engineering jobs__

1. update feature engineering input functions per required parameters
2. run below function to start the feature engineering job
```console
from eks_ml_pipeline import node_autoencoder_fe_input, node_fe_pipeline
node_fe_pipeline(*node_autoencoder_fe_input())
```

### __Using s3_utilities__
s3_utilities has a number of helper functions for the pipeline to download and upload files/objects to s3.
#### - Usage
Import
```console
from eks_ml_pipeline import S3Utilities
```
Class is initilized with the following three parameters
```console
bucket_name = "example_bucket"
model_name = "example_autoencoder"
version = "v0.0.1"
S3Utills = S3Utilities(bucket_name,model_name,version)
```
The following functions can be accessed through the class
```console
1. S3Utills.upload_file(local_path, bucket_name, key)
2. S3Utills.download_file(local_path, bucket_name, key)
3. S3Utills.download_zip(writing_path, folder, type_, file_name)
4. S3Utills.unzip(path_to_zip, extract_location)
5. S3Utills.zip_and_upload(local_path, folder, type_, file_name)
6. S3Utills.pandas_dataframe_to_s3(input_datafame, folder, type_, file_name)
7. S3Utills.write_tensor(tensor, folder, type_, file_name)
8. S3Utills.awswrangler_pandas_dataframe_to_s3(input_datafame,folder, type_, file_name)
9. S3Utills.read_tensor(folder, type_, file_name)
10. S3Utills.upload_directory(local_path, folder, type_)
11. S3Utills.pyspark_write_parquet(df,folder, type_)
12. S3Utills.read_parquet_to_pandas_df(folder, type_, file_name)
```
Note: More helper functions can be added in the future without changing <br>
the structure of the class new functions can just be appened to the class. 
### __s3 structure__
This is the example s3 structure enforced by the s3_utilities class.
All the important variables to note:
```console
bucket_name = "example_bucket"
model_name = "example_autoencoder"
version = "v0.0.1"
folder = "data" or ""models"
type_ =  "pandas_df" or "tensors" or "zipped_models", "npy_models"
file_name = "training_2022_10_10_10.parquet" 
```
The following structure will be created when the pipeline is run in ```example_bucket```.
```
example_bucket
├── example_autoencorder
│├── v0.0.1
││└── data
││    ├── pandas_df
││    │└── training_2022_10_10_10.parquet
││    └── tensors
││        └── training_2022_10_10_10.npy
│└── v0.0.2
│    ├── data
│    │├── pandas_df
│    ││├── testing_2022_9_29.parquet
│    ││└── training_2022_9_29.parquet
│    │└── tensors
│    │    ├── testing_2022_9_29.npy
│    │    ├── testing_2022_9_29_1.npy
│    │    ├── training_2022_9_29.npy
│    │    └── training_2022_9_29_1.npy
│    └── models
│        └── onnx_models
│            └── pod_autoencoder_ad_model_v0.0.1-test_training_2022_9_9_1.onnx
│        └── zipped_models
│            └── pod_autoencoder_ad_model_v0.0.1-test_training_2022_9_9_1.zip
│        └── predictions
│            └── testing_2022_9_29_1_predictions.npy
│            └── testing_2022_9_29_1_residuals.npy
│            └── inference_pod_id_40f6b928-9ac6-4824-9031-a52f5d529940_predictions.npy
│            └── inference_pod_id_40f6b928-9ac6-4824-9031-a52f5d529940_residuals.npy

```

### __Running Model Training jobs__

1. update model training input functions per required parameters (eks_ml_pipeline/inputs/training_input.py)
2. run below function to start the model training job
```console
from eks_ml_pipeline import model_training_pipeline
from eks_ml_pipeline import node_pca_input, pod_pca_input, container_pca_input
from eks_ml_pipeline import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input

###***Autoencoder***###

#Train node autoencoder model and save on s3
model_training_pipeline(*node_autoencoder_input())

#Train pod autoencoder model and save on s3
model_training_pipeline(*pod_autoencoder_input())

#Train container autoencoder model and save on s3
model_training_pipeline(*container_autoencoder_input())

###***PCA***###

#Train node pca model and save on s3
model_training_pipeline(*node_pca_input())

#Train pod pca model and save on s3
model_training_pipeline(*pod_pca_input())

#Train container pca model and save on s3
model_training_pipeline(*container_pca_input())

```

### __Running Model Evaluation/Testing jobs__

1. update model training input functions per required parameters (eks_ml_pipeline/inputs/training_input.py)
2. run below function to start the model training job
```console
from eks_ml_pipeline import model_evaluation_pipeline
from eks_ml_pipeline import node_pca_input, pod_pca_input, container_pca_input
from eks_ml_pipeline import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input

##***Autoencoder***###

#Test node autoencoder model and save on s3
model_evaluation_pipeline(*node_autoencoder_input())

#Test pod autoencoder model and save on s3
model_evaluation_pipeline(*pod_autoencoder_input())

#Test container autoencoder model and save on s3
model_evaluation_pipeline(*container_autoencoder_input())

##***PCA***###

#Test node pca model and save on s3
model_evaluation_pipeline(*node_pca_input())

#Test pod pca model and save on s3
model_evaluation_pipeline(*pod_pca_input())

#Test container pca model and save on s3
model_evaluation_pipeline(*container_pca_input())

```

### __Running Model Inference jobs__

1. update model inference input functions per required parameters (eks_ml_pipeline/inputs/inference_input.py)
2. run below function to start the model training job
```console
from eks_ml_pipeline import inference_pipeline, model_evaluation_pipeline
from eks_ml_pipeline import node_inference_input, pod_inference_input, container_inference_input
from eks_ml_pipeline import node_pca_input, pod_pca_input, container_pca_input
from eks_ml_pipeline import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input

##***Autoencoder***###

#Inference for node autoencoder model
inference_pipeline(node_inference_input(), node_autoencoder_input(), model_evaluation_pipeline)

#Inference for pod autoencoder model
inference_pipeline(pod_inference_input(), pod_autoencoder_input(), model_evaluation_pipeline)

#Inference for container autoencoder model
inference_pipeline(container_inference_input(), container_autoencoder_input(), model_evaluation_pipeline)

###***PCA***###

#Inference for node pca model
inference_pipeline(node_inference_input(), node_pca_input(), model_evaluation_pipeline)

#Inference for pod pca model
inference_pipeline(pod_inference_input(), pod_pca_input(), model_evaluation_pipeline)

#Inference for container pca model
inference_pipeline(container_inference_input(), container_pca_input(), model_evaluation_pipeline)
    
```
>>>>>>> origin
