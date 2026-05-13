namespace Shared.Dtos.RiskAnalysisDto.Response
{
    /// <summary>
    /// Returned immediately (202 Accepted) and also produced inside the job
    /// for logging / future webhook delivery.
    /// </summary>
    public class RiskAnalysisResponse
    {
        public string StudentId             { get; set; } = string.Empty;
        public string AttemptId             { get; set; } = string.Empty;
        public double OriginalViolationRate { get; set; }

        /// <summary>Per-question breakdown.</summary>
        public List<QuestionRiskResult> QuestionsRisk { get; set; } = [];

        /// <summary>
        /// Average of all per-question student risk scores – stored as the
        /// CheatingReport.RiskScore after the job completes.
        /// </summary>
        public decimal OverallSessionRiskScore { get; set; }
    }

    public class QuestionRiskResult
    {
        public int     QuestionId           { get; set; }
        /// <summary>Min-max normalized, weighted score for this student on this question.</summary>
        public decimal StudentRiskScore     { get; set; }
        /// <summary>Average score of all cohort students who answered this question.</summary>
        public decimal CohortAvgRiskScore   { get; set; }
    }
}
