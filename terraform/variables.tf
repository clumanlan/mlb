variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
}

variable "project_name" {
  description = "Project Name matching S3 Bucket"
  type        = string
  default     = "mlbdk"
}