#!/bin/bash
# EC2 first-boot bootstrap for moderation-engine baseline host.
#
# Target AMI: Amazon Linux 2023 (x86_64).
# Idempotent within a single boot; AL2023 cloud-init runs this once.

set -euxo pipefail

# Install and enable Docker.
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user

# Sanity check (lands in /var/log/cloud-init-output.log).
docker --version
