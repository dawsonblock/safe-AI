# Production Deployment Guide

This guide covers deploying SAFE-AI Governor to production environments.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Load Balancer                        │
│                      (nginx/ALB)                         │
└─────────────────────────────────────────────────────────┘
                            │
                            ├── mTLS
                            │
┌─────────────────────────────────────────────────────────┐
│              SAFE-AI Governor (API)                      │
│                                                          │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐            │
│  │   Gate   │  │ Policy VM │  │ Sandbox  │            │
│  └──────────┘  └───────────┘  └──────────┘            │
│                                                          │
│  ┌──────────┐  ┌───────────┐                           │
│  │  Audit   │  │   Keys    │                           │
│  └──────────┘  └───────────┘                           │
└─────────────────────────────────────────────────────────┘
           │                    │
           │                    │
    ┌──────▼──────┐      ┌─────▼──────┐
    │ Audit Log   │      │  Key Store │
    │ (Persistent)│      │   (KMS)    │
    └─────────────┘      └────────────┘
```

## Prerequisites

- Kubernetes 1.25+ or Docker Swarm
- Persistent storage (for audit log)
- KMS (AWS KMS, HashiCorp Vault, etc.)
- Monitoring (Prometheus + Grafana)
- Secrets management
- Certificate authority (for mTLS)

## Configuration

### 1. Secrets Management

**DO NOT** store secrets in environment files or ConfigMaps.

Use Kubernetes Secrets or external secret managers:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: safeai-secrets
type: Opaque
data:
  jwt-secret: <base64-encoded-secret>
  jwt-private-key: <base64-encoded-key>
  jwt-public-key: <base64-encoded-key>
```

### 2. Persistent Storage

Create PersistentVolumeClaim for audit log:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: safeai-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
```

### 3. ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: safeai-config
data:
  policy.yaml: |
    safe_mode: true
    autonomy_tier: beta
    allow_paths:
      - src/
      - tests/
    deny_paths:
      - .git/
      - config/keys/
    max_edit_size_bytes: 65536
    max_total_size_bytes: 524288
```

## Kubernetes Deployment

### Full Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safe-ai-governor
  labels:
    app: safe-ai-governor
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: safe-ai-governor
  template:
    metadata:
      labels:
        app: safe-ai-governor
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: safe-ai-governor
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      initContainers:
      - name: init-keys
        image: safe-ai-governor:1.0.0
        command: ['sh', '-c']
        args:
          - |
            if [ ! -f /data/keys/ed25519.default.sk ]; then
              safeai keys generate --key-id default
            fi
        volumeMounts:
        - name: data
          mountPath: /data
      
      containers:
      - name: governor
        image: safe-ai-governor:1.0.0
        imagePullPolicy: Always
        
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        
        env:
        - name: SAFEAI_ENABLE
          value: "on"
        - name: SAFEAI_DATA_DIR
          value: "/data"
        - name: SAFEAI_LOG_LEVEL
          value: "INFO"
        - name: SAFEAI_JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: safeai-secrets
              key: jwt-secret
        - name: SAFEAI_JWT_ALGORITHM
          value: "RS256"
        - name: SAFEAI_JWT_PUBLIC_KEY
          valueFrom:
            secretKeyRef:
              name: safeai-secrets
              key: jwt-public-key
        - name: SAFEAI_JWT_PRIVATE_KEY
          valueFrom:
            secretKeyRef:
              name: safeai-secrets
              key: jwt-private-key
        
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /app/config
        - name: tmp
          mountPath: /tmp
        
        livenessProbe:
          httpGet:
            path: /livez
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
              - ALL
          readOnlyRootFilesystem: true
      
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: safeai-data
      - name: config
        configMap:
          name: safeai-config
      - name: tmp
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: safe-ai-governor
  labels:
    app: safe-ai-governor
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: safe-ai-governor

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: safe-ai-governor

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: safe-ai-governor
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: safe-ai-governor
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: safe-ai-governor
subjects:
- kind: ServiceAccount
  name: safe-ai-governor
```

### Ingress (with mTLS)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: safe-ai-governor
  annotations:
    nginx.ingress.kubernetes.io/auth-tls-verify-client: "on"
    nginx.ingress.kubernetes.io/auth-tls-secret: "default/ca-secret"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - safeai.example.com
    secretName: safeai-tls
  rules:
  - host: safeai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: safe-ai-governor
            port:
              number: 8000
```

## Docker Compose (for staging)

```yaml
version: '3.8'

services:
  safe-ai-governor:
    build: .
    image: safe-ai-governor:1.0.0
    
    ports:
      - "8000:8000"
    
    environment:
      - SAFEAI_ENABLE=on
      - SAFEAI_DATA_DIR=/data
      - SAFEAI_LOG_LEVEL=INFO
      - SAFEAI_JWT_SECRET=${JWT_SECRET}
    
    volumes:
      - safeai-data:/data
      - ./config:/app/config:ro
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/livez"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    
    restart: unless-stopped
    
    security_opt:
      - no-new-privileges:true
    
    cap_drop:
      - ALL
    
    read_only: true
    
    tmpfs:
      - /tmp

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards:ro

volumes:
  safeai-data:
  prometheus-data:
  grafana-data:
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'safe-ai-governor'
    static_configs:
      - targets: ['safe-ai-governor:8000']
    metrics_path: '/metrics'

rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: safe-ai-governor
    interval: 30s
    rules:
      - alert: KillSwitchActivated
        expr: safeai_kill_switch_active == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Kill switch activated"
          description: "SAFE-AI Governor kill switch is active"
      
      - alert: HighPolicyBlockRate
        expr: rate(safeai_policy_blocks_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of policy blocks"
      
      - alert: SandboxFailures
        expr: rate(safeai_sandbox_executions_total{exit_code!="0"}[5m]) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of sandbox failures"
      
      - alert: APIErrors
        expr: rate(safeai_api_requests_total{status=~"5.."}[5m]) > 1
        for: 5m
        labels:
          severity: error
        annotations:
          summary: "API error rate elevated"
```

## Backup & Disaster Recovery

### Backup Audit Log

```bash
#!/bin/bash
# backup-audit.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/safeai"

# Create backup
kubectl exec -n default safe-ai-governor-xxx -- \
  cat /data/audit.log > "$BACKUP_DIR/audit_${DATE}.log"

# Verify chain
kubectl exec -n default safe-ai-governor-xxx -- \
  safeai audit --verify

# Compress and encrypt
gzip "$BACKUP_DIR/audit_${DATE}.log"
gpg --encrypt --recipient security@example.com \
  "$BACKUP_DIR/audit_${DATE}.log.gz"

# Upload to S3
aws s3 cp "$BACKUP_DIR/audit_${DATE}.log.gz.gpg" \
  s3://backups/safeai/audit/
```

### Restore Procedure

```bash
# Download backup
aws s3 cp s3://backups/safeai/audit/audit_YYYYMMDD.log.gz.gpg .

# Decrypt and decompress
gpg --decrypt audit_YYYYMMDD.log.gz.gpg | gunzip > audit.log

# Copy to pod
kubectl cp audit.log default/safe-ai-governor-xxx:/data/audit.log

# Verify chain
kubectl exec -n default safe-ai-governor-xxx -- \
  safeai audit --verify
```

## Security Hardening

### 1. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: safe-ai-governor
spec:
  podSelector:
    matchLabels:
      app: safe-ai-governor
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: kms
    ports:
    - protocol: TCP
      port: 443
```

### 2. Pod Security Standards

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: safe-ai-governor-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  readOnlyRootFilesystem: true
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'persistentVolumeClaim'
    - 'secret'
```

### 3. RBAC Policies

Minimal RBAC permissions for the service account.

## Performance Tuning

### Database (Audit Log)

For high-throughput environments, consider:
- SSD storage for audit log
- Log rotation with compression
- Archived logs to S3/GCS

### API Scaling

```yaml
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: safe-ai-governor
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: safe-ai-governor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Compliance

- Enable audit logging at load balancer level
- Retain audit logs for required period (typically 1-7 years)
- Implement log forwarding to SIEM
- Regular security scans (Trivy, Snyk)
- Penetration testing schedule

## Support

For production deployment assistance:
- Enterprise support: enterprise@safe-ai.dev
- Consulting: consulting@safe-ai.dev
