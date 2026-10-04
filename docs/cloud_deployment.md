# Cloud Deployment

## Honest scope statement
No AWS account/credentials are available in this environment, so
nothing here was actually `terraform apply`'d or deployed. What follows
IS real, valid infrastructure-as-code (`infra/*.tf`) and real Dockerfiles
that build a deployable image -- verified by hand for correctness, not
by a live `terraform plan`/`apply`. The one thing needing your own
verification: run `terraform validate` and `terraform plan` yourself
against a real AWS account.

## Architecture
```
Internet -> ALB (port 80)
              |-- /  ................. -> ECS Fargate: frontend (Next.js, :3000)
              |-- /fleet/*, /tires/*,
              |   /predict/*, /copilot/*,
              |   /health, /docs ..... -> ECS Fargate: backend (FastAPI, :8000)
                                              |
                                              v
                                        RDS PostgreSQL (private, db.t3.micro)
```
Images built from `Dockerfile.backend` / `Dockerfile.frontend`, pushed
to ECR, run as ECS Fargate services. Secrets (DB URL, Gemini key) live
in Secrets Manager, injected at runtime -- never in the task definition
or image.

## Why each service (and why not others)
- **RDS PostgreSQL** -- direct managed replacement for Milestone 11's
  local Postgres. `db.t3.micro`/20GB gp3 fits the AWS free tier.
- **ECS Fargate** (not EKS) -- 2 services, no need for Kubernetes'
  scheduling complexity. Fargate also means no EC2 instances to patch.
- **ALB** -- single HTTP entrypoint, path-based routing to the 2
  services, no need for 2 separate load balancers.
- **ECR** -- required to store the images ECS pulls; no alternative that
  avoids it.
- **Secrets Manager** -- least-privilege IAM lets the ECS execution role
  read only the 3 secrets this app needs, not `secretsmanager:*`.
- **CloudWatch Logs** -- default ECS logging destination, 7-day
  retention (a portfolio project doesn't need long retention).
- **Rejected: AWS IoT Core.** Milestone 9's MQTT (mosquitto) already
  works for this project's scale. IoT Core earns its cost/complexity at
  real device-fleet scale (device certs, fleet provisioning) -- not
  justified here.
- **Rejected: Lambda + EventBridge.** Nothing in this project currently
  needs scheduled/event-driven compute (training is run manually). Would
  be a real future addition for scheduled retraining, not built now
  since it wouldn't be exercised by anything that exists.
- **Rejected: custom VPC.** Default VPC + security groups are
  sufficient; a custom VPC with public/private subnet split and a NAT
  gateway is the right call at production scale, not before.

## Cost (rough, us-east-1)
Free tier (12mo): RDS db.t3.micro + 20GB, 750 EC2-equivalent hrs.
Fargate is NOT in the free tier: ~$15-20/mo for the 2 small tasks
described here (512+256 CPU units) if left running continuously --
worth stopping services (`desired_count = 0`) when not actively demoing.

## Deploying (for your own AWS account)
```bash
docker build -f Dockerfile.backend -t tireguard-backend .
docker build -f Dockerfile.frontend -t tireguard-frontend .

cd infra
terraform init
terraform validate
terraform plan -var="db_password=..." -var="llm_api_key=..."
terraform apply -var="db_password=..." -var="llm_api_key=..."

# push images to the ECR URLs from `terraform output`, then
# force a new ECS deployment to pick them up.
```

## Known limitations
- No HTTPS/ACM cert or custom domain configured (needs a real domain).
- No autoscaling -- `desired_count = 1` fixed.
- No blue/green deploy -- new images require a manual service update.
- `skip_final_snapshot = true` on RDS -- fine for a portfolio DB, not for
  real production data.