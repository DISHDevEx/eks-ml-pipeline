

def cleanup(feature_list):

    processed_features = []
    for feature in features:
        feature = feature.strip()
        feature = "".join(feature.split())
        feature = feature.replace('.','')
        processed_features.append(feature)

    return processed_features