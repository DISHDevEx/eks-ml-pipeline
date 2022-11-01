Data preprocessing, feature engineering, training and deployment of eks pattern detection models.



# Project Structure

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
├──  reporting				contains all reporting modules 
│    ├── example_roc_report.py
│
│
└── tests						this foler contains unit and integration tests of the project
│    ├── unit
│    └── integration
│ 
└── utilities					any additional utilties we need for the project
│    └── eventually we will be moving to msspackages│  
│ 

```

     
     