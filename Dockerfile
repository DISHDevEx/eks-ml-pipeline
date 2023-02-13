#FROM public.ecr.aws/amazonlinux/amazonlinux:2.0.20221210.1-amd64 AS base
FROM --platform=linux/amd64 amazonlinux:2 AS base
RUN yum install -y python3 git
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
RUN pip3 install git+https://github.com/DISHDevEx/dish-devex-sdk.git
RUN pip3 install git+https://github.com/DISHDevEx/eks-ml-pipeline.git@aakash/emr-pr
#RUN apt install yarn

RUN touch .env
RUN --mount=type=secret,id=BUCKET_NAME_RAW_DATA \
  --mount=type=secret,id=FOLDER_NAME_RAW_DATA \
  export BUCKET_NAME_RAW_DATA=$(cat /run/secrets/BUCKET_NAME_RAW_DATA) && \
  export FOLDER_NAME_RAW_DATA=$(cat /run/secrets/FOLDER_NAME_RAW_DATA) && \
  echo $BUCKET_NAME_RAW_DATA \
  python genenv.py \

COPY . .
RUN pip3 install venv-pack==0.2.0
RUN  pip3 install boto3
RUN  pip3 install pyarrow
RUN  pip3 install awswrangler
RUN  pip3 install fast-arrow
RUN  pip3 install tf2onnx
RUN  pip3 install python-dotenv
RUN pwd

RUN mkdir /output && venv-pack -o /output/pyspark_deps_github.tar.gz

FROM scratch AS export
COPY --from=base /output/pyspark_deps_github.tar.gz /