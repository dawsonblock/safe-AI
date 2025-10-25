"""Prometheus metrics exporter."""

from prometheus_client import Counter, Gauge, Histogram, generate_latest


class MetricsCollector:
    """Collects and exposes Prometheus metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        # Policy decisions
        self.policy_decisions = Counter(
            "safeai_policy_decisions_total",
            "Total policy decisions",
            ["verdict"],
        )

        self.policy_blocks = Counter(
            "safeai_policy_blocks_total",
            "Total policy blocks",
            ["reason"],
        )

        # Signatures
        self.signature_verifications = Counter(
            "safeai_signature_verifications_total",
            "Total signature verifications",
            ["result"],
        )

        # Apply operations
        self.apply_operations = Counter(
            "safeai_apply_operations_total",
            "Total apply operations",
            ["result"],
        )

        self.rollback_operations = Counter(
            "safeai_rollback_operations_total",
            "Total rollback operations",
            ["result"],
        )

        # Sandbox executions
        self.sandbox_executions = Counter(
            "safeai_sandbox_executions_total",
            "Total sandbox executions",
            ["exit_code"],
        )

        self.sandbox_timeouts = Counter(
            "safeai_sandbox_timeouts_total",
            "Total sandbox timeouts",
        )

        self.sandbox_duration = Histogram(
            "safeai_sandbox_duration_seconds",
            "Sandbox execution duration",
        )

        # Audit log
        self.audit_records = Counter(
            "safeai_audit_records_total",
            "Total audit records",
        )

        # Kill switch
        self.kill_switch_active = Gauge(
            "safeai_kill_switch_active",
            "Kill switch status (1=active, 0=inactive)",
        )

        # API requests
        self.api_requests = Counter(
            "safeai_api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"],
        )

        self.api_duration = Histogram(
            "safeai_api_duration_seconds",
            "API request duration",
            ["endpoint", "method"],
        )

    def record_policy_decision(self, verdict: str) -> None:
        """Record policy decision.

        Args:
            verdict: Policy verdict (approved/blocked)
        """
        self.policy_decisions.labels(verdict=verdict).inc()

    def record_policy_block(self, reason: str) -> None:
        """Record policy block.

        Args:
            reason: Block reason
        """
        self.policy_blocks.labels(reason=reason).inc()

    def record_signature_verification(self, success: bool) -> None:
        """Record signature verification.

        Args:
            success: Whether verification succeeded
        """
        result = "success" if success else "failure"
        self.signature_verifications.labels(result=result).inc()

    def record_apply_operation(self, success: bool) -> None:
        """Record apply operation.

        Args:
            success: Whether operation succeeded
        """
        result = "success" if success else "failure"
        self.apply_operations.labels(result=result).inc()

    def record_rollback_operation(self, success: bool) -> None:
        """Record rollback operation.

        Args:
            success: Whether operation succeeded
        """
        result = "success" if success else "failure"
        self.rollback_operations.labels(result=result).inc()

    def record_sandbox_execution(
        self, exit_code: int, duration_seconds: float, timed_out: bool
    ) -> None:
        """Record sandbox execution.

        Args:
            exit_code: Exit code
            duration_seconds: Execution duration
            timed_out: Whether execution timed out
        """
        self.sandbox_executions.labels(exit_code=str(exit_code)).inc()
        self.sandbox_duration.observe(duration_seconds)

        if timed_out:
            self.sandbox_timeouts.inc()

    def record_audit_record(self) -> None:
        """Record audit log entry."""
        self.audit_records.inc()

    def set_kill_switch_status(self, active: bool) -> None:
        """Set kill switch status.

        Args:
            active: Whether kill switch is active
        """
        self.kill_switch_active.set(1 if active else 0)

    def record_api_request(
        self, endpoint: str, method: str, status: int, duration_seconds: float
    ) -> None:
        """Record API request.

        Args:
            endpoint: API endpoint
            method: HTTP method
            status: HTTP status code
            duration_seconds: Request duration
        """
        self.api_requests.labels(
            endpoint=endpoint, method=method, status=str(status)
        ).inc()
        self.api_duration.labels(endpoint=endpoint, method=method).observe(duration_seconds)

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus metrics text
        """
        return generate_latest()


# Global metrics instance
metrics = MetricsCollector()
