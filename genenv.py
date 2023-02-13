import os
def get_config_from_environment(env_content):
    env_content += "BUCKET_NAME_RAW_DATA={}\n".format(os.environ.get("BUCKET_NAME_RAW_DATA", ""))
    env_content += "FOLDER_NAME_RAW_DATA={}\n".format(os.environ.get("FOLDER_NAME_RAW_DATA", ""))
    name1 = os.environ.get("BUCKET_NAME_RAW_DATA", "")
    name2 = os.environ.get("FOLDER_NAME_RAW_DATA", "")
    print(name1)
    print(name2)
    return env_content



env_content = ""
env_content = get_config_from_environment(env_content)
with open(".env.Sample", "w", encoding="utf8") as env:
    env.write(env_content)