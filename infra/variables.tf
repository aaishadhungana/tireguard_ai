variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "tireguard-ai"
}

variable "db_password" {
  description = "RDS master password. Pass via TF_VAR_db_password, never commit it."
  type        = string
  sensitive   = true
}

variable "llm_api_key" {
  description = "Gemini API key, stored in Secrets Manager, never in the task definition."
  type        = string
  sensitive   = true
}
