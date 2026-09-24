# Same glibc reasoning as daily_feature_materialize's Dockerfile:
# python:3.11's Lambda base image runs on Amazon Linux 2 (glibc ~2.26), too
# old for numpy/pyarrow's current manylinux_2_28 wheels. python:3.12 runs on
# Amazon Linux 2023 (glibc 2.34), which has real prebuilt wheels.
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# --no-deps: xgboost's wheel unconditionally requires nvidia-nccl-cu13 on
# Linux (347MB, GPU-only, see requirements.txt's comment) — scipy above is
# its one real CPU runtime dependency, installed separately instead.
RUN pip install xgboost==3.4.1 --no-deps --no-cache-dir

# Flat sibling modules (handler.py's own imports: from predict import ...,
# from model_registry import ..., from status_writer import ...; predict.py's
# own import: from batter_lineup import ...; batter_lineup.py's own import:
# from data_readers import ...) plus feature_store.yaml for
# FeatureStore(repo_path=...) to find, same directory as handler.py.
COPY handler.py predict.py model_registry.py status_writer.py batter_lineup.py data_readers.py feature_store.yaml ${LAMBDA_TASK_ROOT}/

# Nested package, NOT flattened — batter_lineup.py imports this as
# `from data.modules.preprocessing import create_batting_order`
# (package-qualified, not a flat sibling import), so the directory structure
# must be preserved.
COPY data ${LAMBDA_TASK_ROOT}/data

CMD ["handler.lambda_handler"]
