from emr_serverless import EMRServerless
import argparse

def parse_args():
    """
    Prepare the info given with python execution command 
    for passing to run_spark_job
    """
    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group(
        "required named arguments"
        )  # to display hyphen-prefixed args under "required arguments" group
    required_named.add_argument(
        "--job-role-arn", 
        help = "EMR Serverless IAM Job Role ARN", 
        required = False
        )
    required_named.add_argument(
        "--applicationId",
        help = "ApplicationId",
        required = True,
        )
    required_named.add_argument(
        "--s3-bucket",
        help = "S3 Bucket to use for logs and job output",
        required = True,
        )
    required_named.add_argument(
        "--entry-point",
        help = "Entry point to EMR serverless",
        required = True,
        )
    required_named.add_argument(
        "--zipped-env",
        help = ("Path to the custom spark and python environemnt to use, "
                + "with all the dependencies installed"),
        required = True,
        )
    required_named.add_argument(
        "--custom-spark-config",
        help = "Custom spark config",
        required = False,
        )

    return parser.parse_args()

# Require s3 bucket and job role to be provided
args = parse_args()
serverless_job_role_arn = args.job_role_arn
ap_id = args.applicationId
s3_bucket = args.s3_bucket
emr_emtry_point = args.entry_point
zipped_env_path = args.zipped_env
custom_spark_config = args.custom_spark_config

# Create and start a new EMRServerless Spark Application
emr_serverless = EMRServerless()

print("Starting EMR Serverless Spark App")
#     # uncomment below lines of code if you want to create a new application
#     emr_serverless.create_application(
#         "pd-autoencoder-test-emr-cli",
#         "emr-6.6.0")
emr_serverless.start_application(ap_id)
print(emr_serverless)

# Run (and wait for) a Spark job
print("Submitting new Spark job")
job_id = emr_serverless.run_spark_job(
    script_location = emr_emtry_point,
    job_role_arn = serverless_job_role_arn,
    application_id = ap_id,
    arguments = [f"s3://{s3_bucket}/emr_serverless/output"],
    s3_bucket_name = s3_bucket,
    zipped_env_path = zipped_env_path,
    custom_spark_config = custom_spark_config,
    )
job_status = emr_serverless.get_job_run()
print(f"Job finished: {job_id}, status is: {job_status.get('state')}")

# Fetch and print the logs
spark_driver_logs = emr_serverless.fetch_driver_log(s3_bucket)

print("Done!")
