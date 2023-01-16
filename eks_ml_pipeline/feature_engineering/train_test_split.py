def all_rectypes_train_test_split(input_df, split_weights):
    """
    inputs
    ------
            input_df: df
            processed/filtered input df from pre processing
            
    outputs
    -------
            node_train : train df
            node_test: test df
            
    """
    train, test = input_df.randomSplit(weights=split_weights, seed=200)
    return train, test