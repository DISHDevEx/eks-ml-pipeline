Data preprocessing, feature engineering, training and deployment of eks pattern detection models.


### __Project Structure__

Initial project structure for the eks-ml-pipeline. This will evolve over the time to reflect latest changes. 

```

├──  data_prep					contains scripts for preparing train, test, validation and score samples
│   ├── example_prep_train_samp.py 
│
│
├──  feature_engineering		contains all feature engineering modules
│   ├── example_feature_rollup_node.py
│
│
├──  modeling					contains all the models and its artifacts including metrics
│   ├── example_model.py
│
│
├──  notebooks					jupyter notebooks to run end-to-end modeling
│   ├── example_cluster_anomaly_sarima.ipynb
│
│
├──  pre_processing				all preprocessing and transformations go here 
│    ├── example_sparse_to_dense.py  
│
│
├──  reporting					contains all reporting modules 
│    ├── example_roc_report.py
│
│
└── tests						contains unit and integration tests of the project
│    ├── unit
│    └── integration
│ 
└── utilities					any additional utilties we need for the project
│    └── eventually we will be moving to msspackages│  
│ 

```


### __Running EMR Serverless jobs from Sagemaker__

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

##### __Run examples__ 
Navigate to the path where the __emr_serverless.py__ script resides and then run any of the following commands as per your use case: 

__1. With only required arguments__ 
```console
python emr_serverless.py --applicationId 00f66mmuts7enm09 --s3-bucket dish-5g.core.pd.g.dp.eks.logs.e --entry-point s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py --zipped-env s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz
```

__2. With required argumemts and job_role_arn__ 
```console
python emr_serverless.py --job-role-arn arn:aws:iam::064047601590:role/Pattern-Detection-EMR-Serverless-Role --applicationId 00f66mmuts7enm09 --s3-bucket dish-5g.core.pd.g.dp.eks.logs.e --entry-point s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py --zipped-env s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz
```

__3. With required arguments and custom_spark_config__
```console
python emr_serverless.py --applicationId 00f66mmuts7enm09 --s3-bucket dish-5g.core.pd.g.dp.eks.logs.e --entry-point s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py --zipped-env s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz --custom-spark-config "--conf spark.driver.maxResultSize=2g --conf spark.driver.memory=10g --conf spark.executor.cores=4 --conf spark.executor.memory=15g --conf spark.memory.offHeap.size=2g"
```

__4. With all arguments__ 
```console
python emr_serverless.py --job-role-arn arn:aws:iam::064047601590:role/Pattern-Detection-EMR-Serverless-Role --applicationId 00f66mmuts7enm09 --s3-bucket dish-5g.core.pd.g.dp.eks.logs.e --entry-point s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py --zipped-env s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz --custom-spark-config "--conf spark.driver.maxResultSize=2g --conf spark.driver.memory=10g --conf spark.executor.cores=4 --conf spark.executor.memory=15g --conf spark.memory.offHeap.size=2g"
```

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
##### __2.a) When submitting a new job to EMR serverless application__
```console
# id of the existing application to submit jobs to
application_id = '00f66mmuts7enm09' 

# serverless_job_role_arn - only pass it if you want to use a custom one, else comment it out
#serverless_job_role_arn = "<<include_custom_role_arn>>"

# s3 bukcet name where the dependencies, logs and code sits
s3_bucket_name = 'dish-5g.core.pd.g.dp.eks.logs.e'

# Entry point to EMR serverless
emr_entry_point = 's3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py'

# Path to the custom spark and python environemnt to use with all the dependencies installed
zipped_env_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz'
```

```console
# Instantiate class
emr_serverless = EMRServerless()
```
```console
## use this only if you want to create a new application
#application_id = emr_serverless.create_application("pd-autoencoder-test-3", "emr-6.9.0")
```
```console
print("Starting EMR Serverless Spark App")
# Start the application; skips this step automatically if the application is already in 'Started' state
emr_serverless.start_application(application_id)
print(emr_serverless)

# Run (and wait for) a Spark job
print("Submitting new Spark job")
job_run_id = emr_serverless.run_spark_job(
    script_location=emr_entry_point,
    #job_role_arn=serverless_job_role_arn,
    application_id = application_id,
    arguments=[f"s3://{s3_bucket_name}/emr_serverless/output"],
    s3_bucket_name=s3_bucket_name,
    zipped_env_path = zipped_env_path
)
```
```console
# Get the configuration and status of the job which we just submitted 
emr_serverless.get_job_run()

# Get final status of the job
emr_serverless.get_job_run().get('state')
```

```console
# Get final status of the job
emr_serverless.get_job_run().get('state')
```
```console
# Cancel job if needed ; Uncomment as per need
#emr_serverless.cancel_job_run()
#emr_serverless.cancel_job_run(job_run_id) # pass in specific job_run_id which you want to cancel

# Verify the state post cancellation
#emr_serverless.get_job_run().get('state')
```
```console
# Fetch and print the logs
emr_serverless.fetch_driver_log(s3_bucket_name)
```
```console
## use below code only when the application needs to be stopped

#emr_serverless.stop_application(application_id)  # pass in specific application_id which you want to stop
#emr_serverless.stop_application() # if no application id is given, it automatically takes the current application which we started 

## use below code only to delete your CUSTOM applications
## DO NOT use this to delete an existing application created by the admins 

#emr_serverless.delete_application(application_id)   # pass in specific application_id which you want to delete
#emr_serverless.delete_application() # if no application id is given, it automatically takes the current application which we started 
```

##### __2.b) Retrieving info on existing jobs in an application__

```console
application_id = '00f66mmuts7enm09' # '00f66mmuts7enm09'
job_run_id = '00f6ldlnm97t5409' #'00f6fkpig0rlip09'
s3_bucket_name = 'dish-5g.core.pd.g.dp.eks.logs.e'
```

```console
# Instantiate class
emr_with_existing_job = EMRServerless(application_id, job_run_id)

# Get job status
emr_with_existing_job.get_job_run()

# Get logs
emr_with_existing_job.fetch_driver_log(s3_bucket_name)
```
