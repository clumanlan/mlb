# terraform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}


# IAM role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# IAM policy attachments
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3_access" {
  name = "${var.project_name}-lambda-s3-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::${var.project_name}",
        "arn:aws:s3:::${var.project_name}/*"
      ]
    }]
  })
}

# SSM permissions for odds API key and quota counter
resource "aws_iam_role_policy" "lambda_ssm_odds" {
  name = "${var.project_name}-lambda-ssm-odds-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ssm:GetParameter",
        "ssm:PutParameter"
      ]
      Resource = [
        "arn:aws:ssm:${var.aws_region}:*:parameter/mlb/odds-api/*"
      ]
    }]
  })
}

# Lambda 1: Daily data fetch
resource "aws_lambda_function" "data_fetch" {
  filename      = "../dist/daily_mlb_fetch.zip"
  function_name = "daily_mlb_fetch"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
  memory_size   = 1024
  lifecycle {
  ignore_changes = [filename, source_code_hash]
  }
  layers = [
    "arn:aws:lambda:us-east-2:336392948345:layer:AWSSDKPandas-Python311:26",
    "arn:aws:lambda:us-east-2:685464669237:layer:mlb-custom:1"
  ]
}

# Lambda 2: Daily data processing
resource "aws_lambda_function" "data_process" {
  filename      = "../dist/daily_process_data.zip"
  function_name = "daily_process_data"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
  memory_size   = 1024
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
  layers = [
    "arn:aws:lambda:us-east-2:336392948345:layer:AWSSDKPandas-Python311:26",
    "arn:aws:lambda:us-east-2:685464669237:layer:mlb-custom:1"
  ]
}

# SQS Dead Letter Queue for failed odds_fetch invocations
resource "aws_sqs_queue" "odds_dlq" {
  name                      = "${var.project_name}-odds-dlq"
  message_retention_seconds = 1209600
}

resource "aws_iam_role_policy" "lambda_sqs_dlq" {
  name = "${var.project_name}-lambda-sqs-dlq-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "sqs:SendMessage"
      Resource = aws_sqs_queue.odds_dlq.arn
    }]
  })
}

# Lambda: Daily schedule fetch (runs at 7 AM EST / 12 UTC, before main pipeline)
resource "aws_lambda_function" "schedule_fetch" {
  filename      = "../dist/daily_schedule_fetch.zip"
  function_name = "daily_schedule_fetch"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.handler"
  runtime       = "python3.11"
  timeout       = 60
  memory_size   = 256
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
  layers = [
    "arn:aws:lambda:us-east-2:685464669237:layer:mlb-custom:1"
  ]
  environment {
    variables = {
      LINEUP_EVENTBRIDGE_RULE_NAME = "${var.project_name}-daily-lineup-fetch"
    }
  }
}

resource "aws_iam_role_policy" "lambda_events_schedule_fetch" {
  name = "${var.project_name}-lambda-events-schedule-fetch-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "events:EnableRule"
      Resource = "arn:aws:events:${var.aws_region}:*:rule/${var.project_name}-daily-lineup-fetch"
    }]
  })
}

resource "aws_cloudwatch_event_rule" "schedule_fetch_schedule" {
  name                = "${var.project_name}-daily-schedule-fetch"
  description         = "Trigger daily_schedule_fetch at 7 AM EST (12 UTC)"
  schedule_expression = "cron(0 12 * * ? *)"
}

resource "aws_cloudwatch_event_target" "schedule_fetch_target" {
  rule      = aws_cloudwatch_event_rule.schedule_fetch_schedule.name
  target_id = "ScheduleFetchLambda"
  arn       = aws_lambda_function.schedule_fetch.arn
}

resource "aws_lambda_permission" "allow_eventbridge_schedule_fetch" {
  statement_id  = "AllowEventBridgeInvokeScheduleFetch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.schedule_fetch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.schedule_fetch_schedule.arn
}

# Lambda 3: Daily odds fetch
resource "aws_lambda_function" "odds_fetch" {
  filename      = "../dist/daily_odds_fetch.zip"
  function_name = "daily_odds_fetch"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
  layers = [
    "arn:aws:lambda:us-east-2:336392948345:layer:AWSSDKPandas-Python311:26",
    "arn:aws:lambda:us-east-2:685464669237:layer:mlb-custom:1"
  ]
  dead_letter_config {
    target_arn = aws_sqs_queue.odds_dlq.arn
  }
}

# IAM role for Step Functions to invoke both Lambdas
resource "aws_iam_role" "sfn_role" {
  name = "${var.project_name}-sfn-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "states.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "sfn_lambda_invoke" {
  name = "${var.project_name}-sfn-lambda-policy"
  role = aws_iam_role.sfn_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "lambda:InvokeFunction"
      Resource = [
        aws_lambda_function.data_fetch.arn,
        aws_lambda_function.data_process.arn
      ]
    }]
  })
}

# Step Functions state machine: fetch then process
resource "aws_sfn_state_machine" "daily_pipeline" {
  name     = "${var.project_name}-daily-pipeline"
  role_arn = aws_iam_role.sfn_role.arn

  definition = jsonencode({
    Comment = "Fetch MLB data then process it"
    StartAt = "FetchData"
    States = {
      FetchData = {
        Type       = "Task"
        Resource   = aws_lambda_function.data_fetch.arn
        ResultPath = "$.fetchResult"
        Next       = "ProcessData"
      }
      ProcessData = {
        Type       = "Task"
        Resource   = aws_lambda_function.data_process.arn
        ResultPath = "$.processResult"
        End        = true
      }
    }
  })
}

# IAM role for EventBridge to start Step Functions executions
resource "aws_iam_role" "eventbridge_sfn_role" {
  name = "${var.project_name}-eventbridge-sfn-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "events.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "eventbridge_sfn_start" {
  name = "${var.project_name}-eventbridge-sfn-policy"
  role = aws_iam_role.eventbridge_sfn_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "states:StartExecution"
      Resource = aws_sfn_state_machine.daily_pipeline.arn
    }]
  })
}

# EventBridge to trigger the daily pipeline at 4 AM EST
resource "aws_cloudwatch_event_rule" "daily_fetch" {
  name                = "${var.project_name}-daily-fetch"
  description         = "Trigger daily data fetch at 6 AM CST"
  schedule_expression = "cron(0 10 * * ? *)"  # 6 AM CST = 10 AM UTC
}

resource "aws_cloudwatch_event_target" "sfn_target" {
  rule      = aws_cloudwatch_event_rule.daily_fetch.name
  target_id = "TriggerSfnTarget"
  arn       = aws_sfn_state_machine.daily_pipeline.arn
  role_arn  = aws_iam_role.eventbridge_sfn_role.arn
}

# SNS topic + email subscription for Lambda alerts
resource "aws_sns_topic" "lambda_alerts" {
  name = "${var.project_name}-lambda-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.lambda_alerts.arn
  protocol  = "email"
  endpoint  = "thealconomist@gmail.com"
}

resource "aws_cloudwatch_metric_alarm" "odds_errors" {
  alarm_name          = "${var.project_name}-odds-fetch-errors"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 60
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Fires when daily_odds_fetch has at least one error"
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = aws_lambda_function.odds_fetch.function_name
  }

  alarm_actions = [aws_sns_topic.lambda_alerts.arn]
  ok_actions    = [aws_sns_topic.lambda_alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "odds_zero_invocations" {
  alarm_name          = "${var.project_name}-odds-fetch-not-invoked"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocations"
  namespace           = "AWS/Lambda"
  period              = 86400
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Fires if daily_odds_fetch had zero invocations in the last 24 hours (expected by 10 AM CST)"
  treat_missing_data  = "breaching"

  dimensions = {
    FunctionName = aws_lambda_function.odds_fetch.function_name
  }

  alarm_actions = [aws_sns_topic.lambda_alerts.arn]
}

resource "aws_cloudwatch_event_rule" "odds_fetch_schedule" {
  name                = "${var.project_name}-daily-odds-fetch"
  description         = "Trigger daily odds fetch Lambda"
  schedule_expression = "cron(0 14 * * ? *)"  # Pick your time — this is 9 AM CST
}

resource "aws_cloudwatch_event_target" "odds_fetch_target" {
  rule      = aws_cloudwatch_event_rule.odds_fetch_schedule.name
  target_id = "OddsFetchLambda"
  arn       = aws_lambda_function.odds_fetch.arn
}

resource "aws_lambda_permission" "allow_eventbridge_odds_fetch" {
  statement_id  = "AllowEventBridgeInvokeOddsFetch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_fetch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.odds_fetch_schedule.arn
}

# Lambda: Daily lineup fetch (enabled dynamically by schedule_fetch)
resource "aws_lambda_function" "lineup_fetch" {
  filename      = "../dist/daily_lineup_fetch.zip"
  function_name = "daily_lineup_fetch"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.handler"
  runtime       = "python3.11"
  timeout       = 60
  memory_size   = 256
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
  layers = [
    "arn:aws:lambda:us-east-2:685464669237:layer:mlb-custom:1"
  ]
  environment {
    variables = {
      LINEUP_EVENTBRIDGE_RULE_NAME = "${var.project_name}-daily-lineup-fetch"
      HARD_CUTOFF_MINUTES          = "30"
    }
  }
}

resource "aws_cloudwatch_event_rule" "lineup_fetch_schedule" {
  name                = "${var.project_name}-daily-lineup-fetch"
  description         = "Polls for confirmed lineups every 15 min — enabled by schedule_fetch, disabled by lineup_fetch"
  schedule_expression = "rate(15 minutes)"
  state               = "DISABLED"
}

resource "aws_cloudwatch_event_target" "lineup_fetch_target" {
  rule      = aws_cloudwatch_event_rule.lineup_fetch_schedule.name
  target_id = "LineupFetchLambda"
  arn       = aws_lambda_function.lineup_fetch.arn
}

resource "aws_lambda_permission" "allow_eventbridge_lineup_fetch" {
  statement_id  = "AllowEventBridgeInvokeLineupFetch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lineup_fetch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.lineup_fetch_schedule.arn
}

# IAM: allow lineup_fetch to disable its own EventBridge rule
resource "aws_iam_role_policy" "lambda_events_lineup_fetch" {
  name = "${var.project_name}-lambda-events-lineup-fetch-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "events:DisableRule"
      Resource = "arn:aws:events:${var.aws_region}:*:rule/${var.project_name}-daily-lineup-fetch"
    }]
  })
}

# Lambda: Daily DraftKings slate fetch
resource "aws_lambda_function" "dkslate_fetch" {
  filename      = "../dist/daily_dkslate_fetch.zip"
  function_name = "daily_dkslate_fetch"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 512
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
  layers = [
    "arn:aws:lambda:us-east-2:685464669237:layer:mlb-custom:1"
  ]
}

resource "aws_cloudwatch_event_rule" "dkslate_fetch_schedule" {
  name                = "${var.project_name}-daily-dkslate-fetch"
  description         = "Daily DraftKings slate fetch at 10 AM CST (4 PM UTC)"
  schedule_expression = "cron(0 16 * * ? *)"
}

resource "aws_cloudwatch_event_target" "dkslate_fetch_target" {
  rule      = aws_cloudwatch_event_rule.dkslate_fetch_schedule.name
  target_id = "DKSlateFetchLambda"
  arn       = aws_lambda_function.dkslate_fetch.arn
}

resource "aws_lambda_permission" "allow_eventbridge_dkslate_fetch" {
  statement_id  = "AllowEventBridgeInvokeDKSlateFetch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dkslate_fetch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.dkslate_fetch_schedule.arn
}

# Lambda: daily_feature_create — computes batter_lineup/batter_pa_volume feature
# snapshots and writes them to S3 (the Feast offline store). No feast import, so
# it stays a normal zip+layers Lambda like everything else above. The
# AWSSDKPandas-Python311 layer bundles pandas AND awswrangler together (formerly
# "aws-data-wrangler"), which is the real dependency data_readers.py needs.
resource "aws_lambda_function" "feature_create" {
  filename      = "../dist/daily_feature_create.zip"
  function_name = "daily_feature_create"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.11"
  # 512MB/300s was too small for real production data: a live smoke test
  # (2026-09-18) had batter_pa_volume's full-season playbyplay/boxscore/
  # schedule read + rolling-window computation hit both the memory ceiling
  # (Max Memory Used: 512 MB, exactly the limit) and the timeout
  # (Status: timeout at the full 300000ms) — a hard platform-level kill, so
  # the invocation never reached the code that invokes
  # daily_feature_materialize. playbyplay alone is ~137MB compressed across
  # 174 files this season; pandas joins/rolling windows multiply that
  # several times over in memory. Sized up rather than guessing
  # incrementally — Lambda's CPU scales with memory, so this also buys more
  # compute, not just headroom.
  timeout       = 900
  memory_size   = 2048
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
  layers = [
    "arn:aws:lambda:us-east-2:336392948345:layer:AWSSDKPandas-Python311:26"
  ]
}

# S3 -> daily_feature_create: fires on every real per-game lineup write
# daily_lineup_fetch makes (raw_data/games/lineups/{year}/{date}/{game_pk}.json).
# Chosen over a fixed daily cron because day/night games confirm lineups at very
# different real times — see DECISIONS.md's 2026-09-17 entries. The mlbdk bucket
# has no other aws_s3_bucket_notification anywhere in this file (confirmed via
# grep before adding this — only one such resource can exist per bucket, and a
# second one would silently replace rather than merge with an existing config).
# daily_lineup_fetch also writes a states-tracking file under this same prefix
# on every poll cycle regardless of whether anything new confirmed
# (raw_data/games/lineups/states/{year}/{date}.json) — S3's prefix/suffix-only
# filters can't exclude just that subpath, so daily_feature_create's own
# handler detects and skips those events (see _date_from_event).
resource "aws_lambda_permission" "allow_s3_invoke_feature_create" {
  statement_id  = "AllowS3InvokeFeatureCreate"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.feature_create.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = "arn:aws:s3:::${var.project_name}"
}

resource "aws_s3_bucket_notification" "feature_create_on_lineup_write" {
  bucket = var.project_name

  lambda_function {
    lambda_function_arn = aws_lambda_function.feature_create.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw_data/games/lineups/"
    filter_suffix       = ".json"
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke_feature_create]
}

# daily_feature_create -> daily_feature_materialize: direct async invoke on
# success (InvocationType="Event" in the handler), not Step Functions — a
# simple two-hop chain doesn't need the state-machine infra daily_pipeline
# (data_fetch -> data_process) uses for its more complex flow.
resource "aws_iam_role_policy" "lambda_invoke_feature_materialize" {
  name = "${var.project_name}-lambda-invoke-feature-materialize-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "lambda:InvokeFunction"
      Resource = aws_lambda_function.feature_materialize.arn
    }]
  })
}

# ECR repo + container-image Lambda: daily_feature_materialize — the only place
# feast gets imported. feast[aws]'s real runtime dependency closure doesn't fit
# Lambda's 250MB zip+layers cap (confirmed 2026-09-17: ~470MB full install), so
# this one Lambda uses package_type=Image instead (up to 10GB). Requires an
# image already pushed to this repo's `latest` tag before the first `apply` —
# not done as part of writing this resource; see DECISIONS.md's 2026-09-17 entry.
resource "aws_ecr_repository" "feature_materialize" {
  name                 = "${var.project_name}-daily-feature-materialize"
  image_tag_mutability = "MUTABLE"
}

# Creating the ECR repo does not by itself let the Lambda *service* pull from
# it — that's a separate resource-based policy on the repo, distinct from any
# IAM user/role permissions. Without this, aws_lambda_function.feature_materialize
# fails to create with "AccessDeniedException: Lambda does not have permission
# to access the ECR image" (confirmed real 2026-09-17). Function ARN is built
# from known values rather than aws_lambda_function.feature_materialize.arn to
# avoid a dependency cycle (the Lambda needs this policy to exist first).
resource "aws_ecr_repository_policy" "feature_materialize" {
  repository = aws_ecr_repository.feature_materialize.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "LambdaECRImageRetrievalPolicy"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action = [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ]
      Condition = {
        StringLike = {
          "aws:sourceArn" = "arn:aws:lambda:${var.aws_region}:685464669237:function:daily_feature_materialize"
        }
      }
    }]
  })
}

resource "aws_lambda_function" "feature_materialize" {
  depends_on    = [aws_ecr_repository_policy.feature_materialize]
  function_name = "daily_feature_materialize"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.feature_materialize.repository_url}:latest"
  timeout       = 300
  memory_size   = 1024
  # Image was built on Apple Silicon (Docker's native default), so it's
  # arm64 — Lambda defaults to x86_64 if this isn't set, which produces a
  # real "Runtime.InvalidEntrypoint: ProcessSpawnFailed" error at invoke
  # time (confirmed 2026-09-17), not a build-time or plan-time failure.
  architectures = ["arm64"]
  lifecycle {
    ignore_changes = [image_uri]
  }
}

# materialize_incremental() writes to Feast's DynamoDB online store —
# aws_iam_role_policy.lambda_s3_access (above) already covers the S3 registry
# and offline-store reads; this adds the DynamoDB half, scoped to this
# project's feast tables (named "{feast project}.{feature_view}", e.g.
# mlb_predictions.batter_lineup_fv — see DECISIONS.md's 2026-09-15 entry).
resource "aws_iam_role_policy" "lambda_dynamodb_feast" {
  name = "${var.project_name}-lambda-dynamodb-feast-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:BatchWriteItem",
        "dynamodb:DescribeTable"
      ]
      Resource = [
        "arn:aws:dynamodb:${var.aws_region}:*:table/mlb_predictions.*"
      ]
    }]
  })
}