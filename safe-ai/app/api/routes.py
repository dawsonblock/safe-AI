"""FastAPI routes for SAFE-AI Governor."""

import time
from datetime import datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import PlainTextResponse

from app.core.apply import ApplyError, FileApplicator
from app.core.audit import AuditLog
from app.core.gate import GateError, ManifestGate
from app.core.kill import KillSwitch
from app.core.logging_config import get_logger, setup_logging
from app.core.metrics import metrics
from app.core.models import (
    ApplyRequest,
    ApplyResponse,
    AuditRecord,
    ChangeManifest,
    PolicyVerdict,
    RollbackRequest,
    RollbackResponse,
    SandboxResult,
    SignedEnvelope,
)
from app.core.policy_vm import PolicyVM
from app.core.sandbox import SandboxRunner

from .deps import (
    check_kill_switch,
    get_audit_log,
    get_file_applicator,
    get_kill_switch,
    get_manifest_gate,
    get_policy_vm,
    get_sandbox_runner,
    require_operator,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SAFE-AI Governor",
    description="Cryptographically-gated AI change management",
    version="1.0.0",
)


@app.get("/livez")
async def liveness() -> dict[str, str]:
    """Liveness probe.

    Returns:
        Status dictionary
    """
    return {"status": "alive"}


@app.get("/readyz")
async def readiness(
    kill_switch: Annotated[KillSwitch, Depends(get_kill_switch)],
) -> dict[str, Any]:
    """Readiness probe.

    Args:
        kill_switch: KillSwitch instance

    Returns:
        Status dictionary
    """
    kill_status = kill_switch.get_status()

    return {
        "status": "ready" if not kill_status["active"] else "not_ready",
        "kill_switch": kill_status,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> bytes:
    """Prometheus metrics endpoint.

    Returns:
        Prometheus metrics
    """
    return metrics.export_metrics()


@app.post("/v1/verify", dependencies=[Depends(require_operator)])
async def verify_manifest(
    envelope: SignedEnvelope,
    gate: Annotated[ManifestGate, Depends(get_manifest_gate)],
) -> dict[str, Any]:
    """Verify signed manifest.

    Args:
        envelope: SignedEnvelope to verify
        gate: ManifestGate instance

    Returns:
        Verification result

    Raises:
        HTTPException: If verification fails
    """
    start_time = time.time()

    try:
        manifest = gate.verify_envelope(envelope)
        manifest_hash = gate.hash_manifest(manifest)

        metrics.record_signature_verification(True)

        logger.info(
            "Manifest verified",
            extra={"manifest_hash": manifest_hash, "targets": manifest.targets},
        )

        duration = time.time() - start_time
        metrics.record_api_request("/v1/verify", "POST", 200, duration)

        return {
            "valid": True,
            "manifest_hash": manifest_hash,
            "targets": manifest.targets,
            "rationale": manifest.rationale,
        }

    except GateError as e:
        metrics.record_signature_verification(False)
        logger.warning("Signature verification failed", extra={"error": str(e)})

        duration = time.time() - start_time
        metrics.record_api_request("/v1/verify", "POST", 400, duration)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Verification failed: {e}",
        )


@app.post("/v1/policy/plan", dependencies=[Depends(require_operator)])
async def plan_policy(
    manifest: ChangeManifest,
    policy_vm: Annotated[PolicyVM, Depends(get_policy_vm)],
) -> dict[str, Any]:
    """Evaluate manifest against policy.

    Args:
        manifest: ChangeManifest to evaluate
        policy_vm: PolicyVM instance

    Returns:
        Policy evaluation result
    """
    start_time = time.time()

    result = policy_vm.evaluate(manifest)

    metrics.record_policy_decision(result.verdict.value)

    for reason in result.reasons:
        metrics.record_policy_block(reason[:50])  # Truncate for label

    logger.info(
        "Policy evaluation",
        extra={
            "verdict": result.verdict.value,
            "targets": manifest.targets,
            "reasons_count": len(result.reasons),
        },
    )

    duration = time.time() - start_time
    metrics.record_api_request("/v1/policy/plan", "POST", 200, duration)

    return {
        "verdict": result.verdict.value,
        "reasons": result.reasons,
        "blocked_paths": result.blocked_paths,
        "blocked_patterns": result.blocked_patterns,
    }


@app.post(
    "/v1/apply",
    dependencies=[Depends(require_operator), Depends(check_kill_switch)],
)
async def apply_change(
    request: ApplyRequest,
    gate: Annotated[ManifestGate, Depends(get_manifest_gate)],
    policy_vm: Annotated[PolicyVM, Depends(get_policy_vm)],
    sandbox: Annotated[SandboxRunner, Depends(get_sandbox_runner)],
    applicator: Annotated[FileApplicator, Depends(get_file_applicator)],
    audit_log: Annotated[AuditLog, Depends(get_audit_log)],
) -> ApplyResponse:
    """Apply signed change.

    Args:
        request: ApplyRequest
        gate: ManifestGate instance
        policy_vm: PolicyVM instance
        sandbox: SandboxRunner instance
        applicator: FileApplicator instance
        audit_log: AuditLog instance

    Returns:
        ApplyResponse

    Raises:
        HTTPException: If apply fails
    """
    start_time = time.time()
    sandbox_result: SandboxResult | None = None

    try:
        # Step 1: Verify signature
        manifest = gate.verify_envelope(request.signed_envelope)
        manifest_hash = gate.hash_manifest(manifest)
        metrics.record_signature_verification(True)

        logger.info("Apply started", extra={"manifest_hash": manifest_hash})

        # Step 2: Verify manifest matches
        if not gate.verify_manifest_hash(request.manifest, manifest_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Manifest hash mismatch",
            )

        # Step 3: Policy evaluation
        policy_result = policy_vm.evaluate(manifest)
        metrics.record_policy_decision(policy_result.verdict.value)

        if policy_result.verdict == PolicyVerdict.BLOCKED:
            # Record blocked attempt
            audit_record = AuditRecord(
                actor="api",
                manifest_sha256=manifest_hash,
                policy_verdict=policy_result.verdict,
                merkle_root="",
                prev_root="",
                applied=False,
                error="; ".join(policy_result.reasons),
            )
            audit_index = audit_log.append(audit_record)
            metrics.record_audit_record()

            logger.warning(
                "Apply blocked by policy",
                extra={"manifest_hash": manifest_hash, "reasons": policy_result.reasons},
            )

            duration = time.time() - start_time
            metrics.record_api_request("/v1/apply", "POST", 403, duration)

            return ApplyResponse(
                success=False,
                manifest_hash=manifest_hash,
                merkle_root=audit_log.get_current_root(),
                audit_index=audit_index,
                message="Blocked by policy",
                error="; ".join(policy_result.reasons),
            )

        # Step 4: Run tests in sandbox
        if manifest.tests:
            sandbox_result = sandbox.run_tests(manifest.tests)
            metrics.record_sandbox_execution(
                sandbox_result.rc,
                sandbox_result.duration_ms / 1000.0,
                sandbox_result.timed_out,
            )

            if sandbox_result.rc != 0:
                # Record failed tests
                audit_record = AuditRecord(
                    actor="api",
                    manifest_sha256=manifest_hash,
                    policy_verdict=policy_result.verdict,
                    sandbox_result=sandbox_result,
                    merkle_root="",
                    prev_root="",
                    applied=False,
                    error="Tests failed",
                )
                audit_index = audit_log.append(audit_record)
                metrics.record_audit_record()

                logger.error(
                    "Tests failed",
                    extra={
                        "manifest_hash": manifest_hash,
                        "exit_code": sandbox_result.rc,
                    },
                )

                duration = time.time() - start_time
                metrics.record_api_request("/v1/apply", "POST", 422, duration)

                return ApplyResponse(
                    success=False,
                    manifest_hash=manifest_hash,
                    merkle_root=audit_log.get_current_root(),
                    audit_index=audit_index,
                    message="Tests failed",
                    sandbox_result=sandbox_result,
                    error="Tests failed with exit code " + str(sandbox_result.rc),
                )

        # Step 5: Apply changes atomically
        apply_result = applicator.apply_manifest(manifest)
        applicator.cleanup()

        # Step 6: Record in audit log
        audit_record = AuditRecord(
            actor="api",
            manifest_sha256=manifest_hash,
            policy_verdict=policy_result.verdict,
            sandbox_result=sandbox_result,
            merkle_root="",
            prev_root="",
            applied=True,
        )
        audit_index = audit_log.append(audit_record)
        metrics.record_audit_record()
        metrics.record_apply_operation(True)

        logger.info(
            "Apply successful",
            extra={
                "manifest_hash": manifest_hash,
                "audit_index": audit_index,
                "files_modified": apply_result["files_modified"],
            },
        )

        duration = time.time() - start_time
        metrics.record_api_request("/v1/apply", "POST", 200, duration)

        return ApplyResponse(
            success=True,
            manifest_hash=manifest_hash,
            merkle_root=audit_log.get_current_root(),
            audit_index=audit_index,
            message=f"Applied successfully: {apply_result['files_modified']} files modified",
            sandbox_result=sandbox_result,
        )

    except GateError as e:
        metrics.record_signature_verification(False)
        metrics.record_apply_operation(False)
        logger.error("Gate error", extra={"error": str(e)})

        duration = time.time() - start_time
        metrics.record_api_request("/v1/apply", "POST", 400, duration)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except ApplyError as e:
        metrics.record_apply_operation(False)
        logger.error("Apply error", extra={"error": str(e)})

        duration = time.time() - start_time
        metrics.record_api_request("/v1/apply", "POST", 500, duration)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post(
    "/v1/rollback",
    dependencies=[Depends(require_operator), Depends(check_kill_switch)],
)
async def rollback_change(
    request: RollbackRequest,
    audit_log: Annotated[AuditLog, Depends(get_audit_log)],
    applicator: Annotated[FileApplicator, Depends(get_file_applicator)],
) -> RollbackResponse:
    """Rollback a change.

    Args:
        request: RollbackRequest
        audit_log: AuditLog instance
        applicator: FileApplicator instance

    Returns:
        RollbackResponse

    Raises:
        HTTPException: If rollback fails
    """
    start_time = time.time()

    try:
        # Find record to rollback
        if request.manifest_hash:
            record = audit_log.find_by_manifest_hash(request.manifest_hash)
        elif request.audit_index is not None:
            record = audit_log.get_record_by_index(request.audit_index)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must specify manifest_hash or audit_index",
            )

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Record not found",
            )

        if not record.applied:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Record was not applied, cannot rollback",
            )

        # For now, rollback is a placeholder - would need to store previous state
        logger.warning(
            "Rollback requested",
            extra={"manifest_hash": record.manifest_sha256},
        )

        metrics.record_rollback_operation(True)

        duration = time.time() - start_time
        metrics.record_api_request("/v1/rollback", "POST", 200, duration)

        return RollbackResponse(
            success=True,
            reverted_hash=record.manifest_sha256,
            new_merkle_root=audit_log.get_current_root(),
            message="Rollback completed (placeholder)",
        )

    except Exception as e:
        metrics.record_rollback_operation(False)
        logger.error("Rollback error", extra={"error": str(e)})

        duration = time.time() - start_time
        metrics.record_api_request("/v1/rollback", "POST", 500, duration)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/v1/audit/{n}", dependencies=[Depends(require_operator)])
async def get_audit_records(
    n: int,
    audit_log: Annotated[AuditLog, Depends(get_audit_log)],
) -> dict[str, Any]:
    """Get last N audit records.

    Args:
        n: Number of records to fetch
        audit_log: AuditLog instance

    Returns:
        Audit records
    """
    start_time = time.time()

    records = audit_log.get_records(limit=n)

    duration = time.time() - start_time
    metrics.record_api_request("/v1/audit/{n}", "GET", 200, duration)

    return {
        "count": len(records),
        "records": [r.model_dump(mode="json") for r in records],
    }


@app.get("/v1/audit/verify", dependencies=[Depends(require_operator)])
async def verify_audit_chain(
    audit_log: Annotated[AuditLog, Depends(get_audit_log)],
) -> dict[str, Any]:
    """Verify audit log Merkle chain.

    Args:
        audit_log: AuditLog instance

    Returns:
        Verification result
    """
    start_time = time.time()

    is_valid, error = audit_log.verify_chain()

    duration = time.time() - start_time
    metrics.record_api_request("/v1/audit/verify", "GET", 200, duration)

    return {
        "valid": is_valid,
        "error": error if error else None,
        "current_root": audit_log.get_current_root(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
