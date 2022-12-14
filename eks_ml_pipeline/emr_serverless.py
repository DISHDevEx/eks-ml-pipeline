import gzip
import boto3
import argparse


class EMRServerless:
    """
    An example implementation of running a PySpark job on EMR Serverless.

    This class provides support for creating an EMR Serverless Spark application, running a job,
    fetching driver logs, and shutting the application back down.

    By default, all calls are synchronous in that they wait for the Application to reach the desired state.
    - `create_application` waits for the application to reach the `CREATED` state.
    - `start_application` waits for the `STARTED` state.
    - `stop_application` waits for the `STOPPED state.
    - `run_spark_job` waits until the job is in a terminal state.
    """
    def __init__(self, application_id: str = None, job_run_id: str = None) -> None:
        self.application_id = application_id
        self.s3_log_prefix = "logs"
        self.app_type = "SPARK"  # EMR Serverless also supports jobs of type 'HIVE'
        self.client = boto3.client("emr-serverless")
        self.job_run_id = job_run_id

    def __str__(self):
        return f"EMR Serverless {self.app_type} Application: {self.application_id}"

    def create_application(self, name: str, release_label: str, wait: bool = True):
        """
        Create a new application with the provided name and release_label - the application needs to be started after.
        """
        if self.application_id is not None:
            raise Exception(
                f"Application already created (application_id: `{self.application_id}`)"
            )

        response = self.client.create_application(
            name=name, releaseLabel=release_label, type=self.app_type
        )
        self.application_id = response.get("applicationId")

        app_ready = False
        while wait and not app_ready:
            response = self.client.get_application(applicationId=self.application_id)
            app_ready = response.get("application").get("state") == "CREATED"

    def start_application(self, wait: bool = True) -> None:
        """
        Start the application - by default, wait until the application is started.
        """
        
        if self.application_id is None:
            raise Exception(
                "No application_id - please use creation_application first."
            )

        self.client.start_application(applicationId=self.application_id)

        app_started = False
        while wait and not app_started:
            response = self.client.get_application(applicationId=self.application_id)
            app_started = response.get("application").get("state") == "STARTED"

    def stop_application(self, wait: bool = True) -> None:
        """
        Stop the application - by default, wait until the application is stopped.
        """
        self.client.stop_application(applicationId=self.application_id)

        app_stopped = False
        while wait and not app_stopped:
            response = self.client.get_application(applicationId=self.application_id)
            app_stopped = response.get("application").get("state") == "STOPPED"

    def delete_application(self) -> None:
        """
        Delete the application - it must be stopped first.
        """
        self.client.delete_application(applicationId=self.application_id)

    def run_spark_job(
        self,
        script_location: str,
        job_role_arn: str,
        application_id: str,
        arguments: list(),
        s3_bucket_name: str,
        zipped_env_path:str,
        wait: bool = True,
    ) -> str:
        """
        Runs the Spark job identified by `script_location`. Arguments can also be provided via the `arguments` parameter.

        By default, spark-submit parameters are hard-coded and logs are sent to the provided s3_bucket_name.
        This method is blocking by default until the job is complete.
        """
        response = self.client.start_job_run(
            applicationId=application_id,
            executionRoleArn=job_role_arn,
            jobDriver={
                "sparkSubmit": {
                    "entryPoint": script_location,
                    "sparkSubmitParameters": f"--conf spark.archives={zipped_env_path}#environment --conf spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON=./environment/bin/python --conf spark.emr-serverless.driverEnv.PYSPARK_PYTHON=./environment/bin/python --conf spark.executorEnv.PYSPARK_PYTHON=./environment/bin/python",
                }
            },
            configurationOverrides={
                "monitoringConfiguration": {
                    "s3MonitoringConfiguration": {
                        "logUri": "s3://emr-serverless-output-pd/logs/"
                    }
                }
            },
        )
        self.job_run_id = response.get("jobRunId")

        job_done = False
        while wait and not job_done:
            jr_response = self.get_job_run(job_run_id)
            job_done = jr_response.get("state") in [
                "SUCCESS",
                "FAILED",
                "CANCELLING",
                "CANCELLED",
            ]

        return job_run_id

    def get_job_run(self) -> dict:
        response = self.client.get_job_run(
            applicationId=self.application_id, jobRunId=self.job_run_id
        )
        return response.get("jobRun")

    def fetch_driver_log(
        self, s3_bucket_name: str, log_type: str = "stdout"
    ) -> str:
        """
        Access the specified `log_type` Driver log on S3 and return the full log string.
        """
        s3_client = boto3.client("s3")
        file_location = f"{self.s3_log_prefix}/applications/{self.application_id}/jobs/{self.job_run_id}/SPARK_DRIVER/{log_type}.gz"
        try:
            response = s3_client.get_object(Bucket=s3_bucket_name, Key=file_location)
            file_content = gzip.decompress(response["Body"].read()).decode("utf-8")
        except s3_client.exceptions.NoSuchKey:
            file_content = ""
        return print("File output from stdout.gz:\n----\n", str(file_content), "\n----")


def parse_args():
    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group(
        "required named arguments"
    )  # Workaround to display hyphen-prefixed args under a "required arguments" group
    required_named.add_argument(
        "--job-role-arn", help="EMR Serverless IAM Job Role ARN", required=True
    )
    required_named.add_argument(
        "--applicationId",
        help="ApplicationId",
        required=True,
    )
    required_named.add_argument(
        "--s3-bucket",
        help="Amazon S3 Bucket to use for logs and job output",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Require s3 bucket and job role to be provided
    args = parse_args()
    serverless_job_role_arn = args.job_role_arn
    application_id = args.applicationId
    s3_bucket_name = args.s3_bucket
    

    # Create and start a new EMRServerless Spark Application
    emr_serverless = EMRServerless()

    print("Starting EMR Serverless Spark App")
    #emr_serverless.create_application("pd-autoencoder-test-emr-cli", "emr-6.6.0")
    emr_serverless.start_application(application_id)
    print(emr_serverless)

    # Run (and wait for) a Spark job
    print("Submitting new Spark job")
    job_run_id = emr_serverless.run_spark_job(
        script_location="s3://emr-serverless-output-pd/code/pyspark/pd-autoencoder-ad/s3_test_emr.py",
        job_role_arn=serverless_job_role_arn,
        application_id = application_id,
        arguments=[f"s3://{s3_bucket_name}/emr-serverless/output"],
        s3_bucket_name=s3_bucket_name,
    )
    job_status = emr_serverless.get_job_run(job_run_id)
    print(f"Job finished: {job_run_id}, status is: {job_status.get('state')}")

    # Fetch and print the logs
    spark_driver_logs = emr_serverless.fetch_driver_log(s3_bucket_name, job_run_id)
    print("File output from stdout.gz:\n----\n", spark_driver_logs, "\n----")

    print("Done! 👋")