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