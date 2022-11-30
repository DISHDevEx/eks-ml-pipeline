

def cleanup(feature_list):

    processed_features = []
    for feature in feature_list:
        cleaned_feature = feature.strip()
        cleaned_feature = "".join(cleaned_feature.split())
        cleaned_feature = cleaned_feature.replace('.','')
        processed_features.append(cleaned_feature)

    return processed_features