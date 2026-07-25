
output "lambda_created_data_fetch" {
  description = "Confirms the data ingestion Lambda function was created"
  value       = "Lambda function created: ${aws_lambda_function.data_fetch.function_name}"
}

output "sfn_state_machine_arn" {
  description = "ARN of the daily pipeline state machine"
  value       = aws_sfn_state_machine.daily_pipeline.arn
}
