# Security Policy

## Overview

SAFE-AI Governor implements defense-in-depth for AI-driven code changes. This document describes the security architecture, threat model, and hardening guidelines.

## Threat Model

### Assets Protected

1. **Source code integrity** - Prevent unauthorized modifications
2. **Execution environment** - Prevent malicious code execution
3. **Cryptographic keys** - Protect signing/verification keys
4. **Audit trail** - Ensure immutable change history
5. **System availability** - Prevent DoS and resource exhaustion

### Threat Actors

1. **Compromised AI agent** - AI system with bugs or adversarial inputs
2. **Insider threat** - Malicious developer with partial access
3. **External attacker** - Network-based exploitation
4. **Supply chain attack** - Compromised dependencies

### Attack Vectors

| Vector | Mitigation |
|--------|-----------|
| Unsigned changes | All changes require Ed25519 signature |
| Policy bypass | Multi-layer policy enforcement (path, content, size) |
| Sandbox escape | Rootless containers, no-NET, RO mounts, capability drop |
| Key theft | Keys stored with 0600 permissions, rotation supported |
| Audit tampering | Merkle-chained append-only log, cryptographic integrity |
| DoS via resources | CPU/memory limits, timeouts, rate limiting |
| Privilege escalation | RBAC with JWT, least-privilege roles |
| Kill switch bypass | Multiple checks (env + file), fail-closed |

## Security Controls

### 1. Cryptographic Verification

**Ed25519 Signatures**
- All manifests must be signed with valid Ed25519 key
- Signature verified before any processing
- Public key pinning supported
- Key rotation with archival

**Threat:** Unsigned or tampered manifests
**Control:** Gate rejects any manifest with invalid or missing signature

```python
# Signature verification flow
manifest = gate.verify_envelope(envelope)  # Raises on invalid
manifest_hash = gate.hash_manifest(manifest)
```

### 2. Policy Enforcement

**Multi-layer Rules**
- Path allowlist/denylist
- Content pattern blocking (regex)
- Operation type restrictions
- Size limits (per-edit and total)

**Threat:** Malicious code injection
**Control:** Policy VM blocks dangerous patterns before execution

```yaml
# Blocked patterns (default)
- subprocess.Popen
- rm -rf
- curl http
- eval(
- exec(
- __import__
```

### 3. Sandbox Isolation

**Rootless OCI Containers**
- No root privileges required
- Network disabled by default
- Read-only project mount
- tmpfs for temporary files
- Capability dropping (CAP_DROP ALL)
- Resource limits (CPU, memory)

**Threat:** Code execution attacks
**Control:** Tests run in isolated sandbox, cannot access host

```bash
podman run --rm \
  --network none \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --cpus 1.0 \
  --memory 512m \
  -v /project:/workspace:ro \
  --tmpfs /tmp:rw,noexec,nosuid \
  python:3.12-slim
```

### 4. Audit Integrity

**Merkle-Chained Log**
- Append-only audit log
- Each record linked via Merkle root
- Tamper detection via chain verification
- Immutable history

**Threat:** Audit log manipulation
**Control:** Cryptographic chaining makes tampering detectable

```python
# Merkle chain
merkle_root = SHA256(prev_root || record_json)
```

### 5. Kill Switch

**Emergency Halt**
- Environment variable check: `SAFEAI_ENABLE`
- Sentinel file check: `KILL_SWITCH`
- Fail-closed design (disabled = halt)
- Checked before all operations

**Threat:** Runaway AI or incident
**Control:** Immediate halt of all apply/rollback operations

### 6. Role-Based Access Control

**JWT Authentication**
- Three roles: admin, operator, auditor
- RS256 or HS256 signing
- Token expiration enforced
- Permission checking on all routes

**Threat:** Unauthorized API access
**Control:** All routes require valid JWT with appropriate role

```python
# Role permissions
admin:    [verify, plan, apply, rollback, audit, keys]
operator: [verify, plan, apply, audit]
auditor:  [audit]
```

### 7. Atomic Operations

**File Modifications**
- Atomic writes via O_TMPFILE + linkat
- Backup before changes
- Automatic rollback on failure
- Idempotency keys

**Threat:** Partial/corrupted writes
**Control:** All-or-nothing semantics, automatic recovery

## Hardening Guidelines

### Key Management

1. **Generate strong keys**
   ```bash
   safeai keys generate --key-id production
   chmod 600 data/keys/ed25519.production.sk
   ```

2. **Rotate regularly**
   ```bash
   # Monthly rotation recommended
   safeai keys rotate --key-id production
   ```

3. **Use KMS in production**
   - Store keys in AWS KMS, HashiCorp Vault, etc.
   - Never commit keys to git
   - Use environment variables for production

4. **Separate dev/prod keys**
   - Different keys for each environment
   - Revoke compromised keys immediately

### Network Security

1. **Enable mTLS**
   ```python
   # API with client cert verification
   app.add_middleware(TLSMiddleware, verify_mode=ssl.CERT_REQUIRED)
   ```

2. **Rate limiting**
   ```python
   from fastapi_limiter import FastAPILimiter
   
   @app.post("/v1/apply")
   @limiter.limit("10/minute")
   async def apply_change(): ...
   ```

3. **IP allowlisting**
   ```python
   # Only allow internal network
   ALLOWED_IPS = ["10.0.0.0/8", "172.16.0.0/12"]
   ```

### Container Security

1. **Use minimal base image**
   ```dockerfile
   FROM python:3.12-slim  # Not python:3.12-full
   ```

2. **Run as non-root**
   ```dockerfile
   USER safeai
   ```

3. **Read-only filesystem**
   ```dockerfile
   VOLUME ["/data", "/tmp"]
   # All other paths read-only
   ```

4. **Security scanning**
   ```bash
   trivy image safe-ai-governor:latest
   ```

### Monitoring & Alerting

1. **Prometheus metrics**
   - Track policy blocks: `safeai_policy_blocks_total`
   - Monitor kill switch: `safeai_kill_switch_active`
   - Watch sandbox failures: `safeai_sandbox_executions_total{exit_code!="0"}`

2. **Alert on anomalies**
   ```yaml
   # Alert if kill switch activated
   - alert: KillSwitchActive
     expr: safeai_kill_switch_active == 1
     for: 1m
     labels:
       severity: critical
   
   # Alert on high policy blocks
   - alert: HighPolicyBlocks
     expr: rate(safeai_policy_blocks_total[5m]) > 10
     for: 5m
     labels:
       severity: warning
   ```

3. **Structured logging**
   ```python
   logger.warning(
       "Policy block",
       extra={
           "manifest_hash": manifest_hash,
           "reasons": policy_result.reasons,
           "actor": request.client.host,
       }
   )
   ```

### Incident Response

1. **Activate kill switch**
   ```bash
   touch KILL_SWITCH
   # Or: export SAFEAI_ENABLE=off
   ```

2. **Verify audit chain**
   ```bash
   safeai audit --verify
   ```

3. **Inspect recent changes**
   ```bash
   safeai audit --tail 100 --json > incident_audit.json
   ```

4. **Rotate keys**
   ```bash
   safeai keys rotate --key-id production
   ```

5. **Review policy**
   - Add patterns that should have been blocked
   - Tighten path restrictions
   - Reduce resource limits

## Compliance Considerations

### SOC 2 Type II

- ✅ Access control (RBAC)
- ✅ Audit logging (Merkle chain)
- ✅ Encryption (Ed25519)
- ✅ Change management (gated apply)
- ✅ Monitoring (Prometheus)

### ISO 27001

- ✅ A.9: Access control
- ✅ A.12: Operations security
- ✅ A.14: System acquisition
- ✅ A.16: Incident management

### PCI-DSS

- ✅ Requirement 6: Secure development
- ✅ Requirement 8: Access control
- ✅ Requirement 10: Logging and monitoring

## Known Limitations

1. **Rollback incomplete** - Currently only logs rollback intent; full state restoration not implemented
2. **Key storage** - Local filesystem storage; KMS integration recommended for production
3. **Single-node** - No distributed consensus; deploy as singleton
4. **Signature only** - No encryption of manifests (use TLS for transport)

## Vulnerability Reporting

**DO NOT** open public issues for security vulnerabilities.

Email: security@safe-ai.dev

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested mitigation

We aim to respond within 48 hours and provide fixes within 7 days for critical issues.

## Security Updates

Monitor:
- GitHub Security Advisories
- Dependency vulnerability scans (Dependabot)
- CVE feeds for base images

Subscribe to security announcements: security-announce@safe-ai.dev

## Audit History

| Date | Auditor | Scope | Findings |
|------|---------|-------|----------|
| 2025-01-15 | Internal | Full system | 0 critical, 2 medium |
| TBD | External | Cryptography + Policy | Pending |

## References

- [NIST SP 800-190](https://csrc.nist.gov/publications/detail/sp/800-190/final) - Container Security
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Ed25519 Spec](https://ed25519.cr.yp.to/)
