namespace DomainLayer.Models
{
    /// <summary>
    /// Persists the calculated risk scores for a single question within one attempt.
    /// Child of RiskAssessmentResult (many per session).
    /// </summary>
    public class RiskQuestionResult : BaseEntity<int>
    {
        // ── Question identity ────────────────────────────────────────────────────
        public int QuestionId { get; set; }

        // ── Calculated scores (rounded to 2 decimal places by the job) ──────────
        /// <summary>
        /// Min-max normalized, weighted risk score for this student on this question.
        /// </summary>
        public decimal StudentRiskScore   { get; set; }

        /// <summary>
        /// Average risk score across all cohort students who answered this question.
        /// Acts as a baseline benchmark.
        /// </summary>
        public decimal CohortAvgRiskScore { get; set; }

        // ── Parent reference ─────────────────────────────────────────────────────
        public int                 RiskAssessmentResultId { get; set; }
        public RiskAssessmentResult RiskAssessmentResult  { get; set; } = null!;
    }
}
