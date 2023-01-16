import os

def cleanup(feature_list):
    """
    Input:
    ------
    feature_list: list
    features names
    
    Output:
    -------
    processed_features: list
    cleaned up features
    """

    processed_features = []
    for feature in feature_list:
        cleaned_feature = feature.strip()
        cleaned_feature = "".join(cleaned_feature.split())
        cleaned_feature = cleaned_feature.replace('.','')
        processed_features.append(cleaned_feature)

    return processed_features