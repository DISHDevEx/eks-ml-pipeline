Data preprocessing, feature engineering, training and deployment of eks pattern detection models.



**Project Structure**

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

Example to run EMR Serverless job from Sagemaker

us-east-1 applications: <br>
* pd-autoencoder-ad-v1 : **00f64bef5869kl09**
* pd-autoencoder-ad-v2 : **00f66ohicnjchu09**
* pd-test-s3-writes : **00f66mmuts7enm09**
<br>

us-west-2 applications: <br>
* pd-autoencoder-ad-container-v1  : **00f672mqiak1fp0l**
<br>

**Note**: while launching your job, please make note of the region from where you are running it.
jobs for us-east-1 applications can only be launched from us-east-1 and similarly, jobs for us-west-2 applications can only be launched from us-west-2. 

#### **Usage Scenario 1: From CLI**

Run the following command <br>

**python emr_serverless.py --job-role-arn <<job_role_arn>> --applicationId <<applicationID>> --s3-bucket <<s3_bucket_name>> --entry-point <<emr_entry_point>> --zipped-env <<zipped_env_path>> --custom-spark-config <<custom_spark_config>>**

optional arguments
- **--job-role-arn**    : default value = 'arn:aws:iam::064047601590:role/Pattern-Detection-EMR-Serverless-Role'
- **--custom-spark-config**   : default value = default
    
Without optional arguments : <br>
**python emr_serverless.py --applicationId <<applicationID>> --s3-bucket <<s3_bucket_name>> --entry-point <<emr_entry_point>> --zipped-env <<zipped_env_path>>**

##### **Run examples** <br>

**1. With only required arguments** <br>

python emr_serverless.py --applicationId 00f66mmuts7enm09 --s3-bucket dish-5g.core.pd.g.dp.eks.logs.e --entry-point s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py --zipped-env s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz
<br>

**2. With required argumemts and job_role_arn** <br>
python emr_serverless.py --job-role-arn arn:aws:iam::064047601590:role/Pattern-Detection-EMR-Serverless-Role --applicationId 00f66mmuts7enm09 --s3-bucket dish-5g.core.pd.g.dp.eks.logs.e --entry-point s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py --zipped-env s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz
<br>

**3. With required arguments and custom_spark_config** <br>
python emr_serverless.py --applicationId 00f66mmuts7enm09 --s3-bucket dish-5g.core.pd.g.dp.eks.logs.e --entry-point s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py --zipped-env s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz --custom-spark-config "--conf spark.driver.maxResultSize=2g --conf spark.driver.memory=10g --conf spark.executor.cores=4 --conf spark.executor.memory=15g --conf spark.memory.offHeap.size=2g"
<br>

**4. With all arguments** <br>
python emr_serverless.py --job-role-arn arn:aws:iam::064047601590:role/Pattern-Detection-EMR-Serverless-Role --applicationId 00f66mmuts7enm09 --s3-bucket dish-5g.core.pd.g.dp.eks.logs.e --entry-point s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/entry_point/s3_test_emr.py --zipped-env s3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_test.tar.gz --custom-spark-config "--conf spark.driver.maxResultSize=2g --conf spark.driver.memory=10g --conf spark.executor.cores=4 --conf spark.executor.memory=15g --conf spark.memory.offHeap.size=2g"
