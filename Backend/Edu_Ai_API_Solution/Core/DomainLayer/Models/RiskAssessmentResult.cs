namespace DomainLayer.Models
{
    /// <summary>
    /// Persists the full calculated risk-assessment result for one quiz attempt.
    /// Created by the background job after cohort normalization is complete.
    /// One-to-one with CheatingReport.
    /// </summary>
    public class RiskAssessmentResult : BaseEntity<int>
    {
        // ── Identity ─────────────────────────────────────────────────────────────
        public string StudentId { get; set; } = string.Empty;
        public int    AttemptId { get; set; }

        // ── Session-level results ────────────────────────────────────────────────
        /// <summary>Original violation_rate extracted directly from session_summary.</summary>
        public double  SessionViolationRate     { get; set; }

        /// <summary>Average of all per-question student risk scores for this session.</summary>
        public decimal OverallSessionRiskScore  { get; set; }

        // ── Relationship to CheatingReport ───────────────────────────────────────
        public int           CheatingReportId { get; set; }
        public CheatingReport CheatingReport  { get; set; } = null!;

        // ── Per-question breakdown ───────────────────────────────────────────────
        public ICollection<RiskQuestionResult> Questions { get; set; } = [];
    }
}
