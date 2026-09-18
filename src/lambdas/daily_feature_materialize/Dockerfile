# python:3.11's Lambda base image runs on Amazon Linux 2 (glibc ~2.26), too old
# for numpy/pyarrow's current manylinux_2_28 wheels — falls back to a from-source
# build with no compiler present. python:3.12 runs on Amazon Linux 2023 (glibc
# 2.34), which has real prebuilt wheels for these. A container-image Lambda's
# Python version is independent of this repo's other Lambdas' python3.11 runtime,
# so this is a safe, contained choice.
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY handler.py feature_store.yaml ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
