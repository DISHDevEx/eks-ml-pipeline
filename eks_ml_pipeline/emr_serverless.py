"""
Class for PySpark jobs on EMR Serverless.
"""

import gzip
import boto3


class EMRServerless:
    """
    An example implementation of running a PySpark job on EMR Serverless.

    Provides support for 
    creating an EMR Serverless Spark application, 
    running a job, 
    fetching driver logs, and 
    shutting the application down.

    By default, all calls are synchronous; 
    they wait for the application to reach the desired state.
    - `create_application` waits for the `CREATED` state.
    - `start_application` waits for the `STARTED` state.
    - `stop_application` waits for the `STOPPED state.
    - `run_spark_job` waits until the job is in a terminal state.
    """
    def __init__(
            self, application_id: str = None, job_run_id: str = None) -> None:
        self.application_id = application_id
        self.s3_log_prefix = "emr_serverless/logs"
        self.app_type = "SPARK"  # EMR Serverless also supports 'HIVE'
        self.client = boto3.client("emr-serverless")
        self.job_run_id = job_run_id

    def __str__(self):
        return f"EMR Serverless {self.app_type} Application: {self.application_id}"

    def create_application(
            self, name: str, release_label: str, wait: bool = True):
        """
        Create a new application with the provided name and release_label
        that the application needs to be started after.
        """
        if self.application_id is not None:
            raise Exception(
                "Application already created (application_id: "
                + f"`{self.application_id}`)"
            )

        response = self.client.create_application(
            name=name, releaseLabel=release_label, type=self.app_type
        )
        self.application_id = response.get("applicationId")

        app_ready = False
        while wait and not app_ready:
            response = self.client\
                            .get_application(applicationId=self.application_id)
            app_ready = response.get("application").get("state") == "CREATED"
        return self.application_id

    def start_application(self, application_id, wait: bool = True) -> None:
        """
        Start the application.
        By default, wait until the application is started.
        """
        self.application_id = application_id

        if self.application_id is None:
            raise Exception(
                "No application_id - please use creation_application first."
                )

        self.client.start_application(applicationId=self.application_id)

        app_started = False
        while wait and not app_started:
            response = self.client.\
                get_application(applicationId=self.application_id)
            app_started = response.get("application").get("state") == "STARTED"

    def stop_application(self, application_id=None, wait: bool = True) -> None:
        """
        Stop the application - by default, wait until the application is stopped.
        """
        if application_id is None:
            self.client.stop_application(applicationId=self.application_id)
        else:
            self.client.stop_application(applicationId=application_id)

        app_stopped = False
        while wait and not app_stopped:
            if application_id is None:
                response = self.client.get_application(
                    applicationId=self.application_id)
            else:
                response = self.client.get_application(
                    applicationId=application_id)
            app_stopped = response.get("application").get("state") == "STOPPED"

        print("Successfully stopped app")

    def delete_application(self, application_id=None) -> None:
        """
        Delete the application - it must be stopped first.
        """
        if application_id is None:
            self.client.delete_application(applicationId=self.application_id)
        else:
            self.client.delete_application(applicationId=application_id)
        print("Successfully deleted app")

    def run_spark_job(
            self,
            script_location: str,
            application_id: str,
            s3_bucket_name: str,
            zipped_env_path:str,
            job_role_arn: str = None,
            custom_spark_config: str = None,
            wait: bool = True,
            ) -> str:
        """
        Run the Spark job identified by `script_location`.

        By default, spark-submit parameters are hard-coded and logs are sent to
        the provided s3_bucket_name.
        This method is blocking by default until the job is complete.
        """

        if job_role_arn is None:
            job_role_arn = ('arn:aws:iam::064047601590:role/'
                            +'Pattern-Detection-EMR-Serverless-Role')

        if custom_spark_config is None:
            custom_spark_config=''

        response = self.client.start_job_run(
            applicationId=application_id,
            executionRoleArn=job_role_arn,
            jobDriver={
                "sparkSubmit": {
                    "entryPoint": script_location,
                    "sparkSubmitParameters":
                        (f"--conf spark.archives={zipped_env_path}#environment "
                         + "--conf spark.emr-serverless.driverEnv"
                         + ".PYSPARK_DRIVER_PYTHON=./environment/bin/python "
                         + "--conf spark.emr-serverless.driverEnv"
                         + ".PYSPARK_PYTHON=./environment/bin/python "
                         + "--conf spark.executorEnv.PYSPARK_PYTHON="
                         + f"./environment/bin/python {custom_spark_config}"),
                }
            },
            configurationOverrides={
                "monitoringConfiguration": {
                    "s3MonitoringConfiguration": {
                        "logUri": f"s3://{s3_bucket_name}/emr_serverless/logs/"
                    }
                }
            },
        )
        self.job_run_id = response.get("jobRunId")
        print(f"job id : {self.job_run_id}")

        print(f'Wait policy is currently {wait}.')
        ## To allow flexibility to cancel jobs if needed
        ## feel free to uncomment the following
        # job_done = False
        # while wait and not job_done:
        #     jr_response = self.get_job_run()
        #     job_done = jr_response.get("state") in [
        #         "SUCCESS",
        #         "FAILED",
        #         "CANCELLING",
        #         "CANCELLED",
        #         ]

        return self.job_run_id

    def get_job_run(self) -> dict:
        """
        Get the configuration and status of the job.
        """
#         client = boto3.client("emr-serverless")
        response = self.client.get_job_run(
            applicationId = self.application_id, jobRunId = self.job_run_id)
        return response.get("jobRun")

    def cancel_job_run(self, job_run_id=None) -> dict:
        """
        Discontinue the job if it is running.
        """
        if job_run_id is None:
            response = self.client.cancel_job_run(
                applicationId=self.application_id, jobRunId=self.job_run_id
                )
        else:
            response = self.client.cancel_job_run(
                applicationId=self.application_id, jobRunId=job_run_id)
        print('Successfully canceled job')
        return response.get("jobRun")

    def fetch_driver_log(
            self, s3_bucket_name: str, log_type: str = "stdout") -> str:
        """
        Access the specified `log_type` Driver log on S3
        and return the full log string.
        """
        s3_client = boto3.client("s3")
        file_location = (f"{self.s3_log_prefix}/applications/"
                         + f"{self.application_id}/jobs/{self.job_run_id}/"
                         + f"SPARK_DRIVER/{log_type}.gz")
        try:
            response = s3_client.get_object(
                Bucket=s3_bucket_name, Key=file_location)
            file_content = gzip.decompress(
                response["Body"].read()).decode("utf-8")
        except s3_client.exceptions.NoSuchKey:
            file_content = ""
        return print("File output from stdout.gz:\n----\n", str(file_content),
                     "\n----")
