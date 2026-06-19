namespace Shared.Dtos.CheatingReportDto.Response
{
    /// <summary>
    /// Per-question risk breakdown returned inside RiskAssessmentResultResponse.
    /// </summary>
    public class RiskQuestionResultDto
    {
        public int     QuestionId         { get; set; }
        public decimal StudentRiskScore   { get; set; }
        public decimal CohortAvgRiskScore { get; set; }
    }

    /// <summary>
    /// Full risk assessment result for a cheating report,
    /// including the per-question breakdown.
    /// </summary>
    public class RiskAssessmentResultResponse
    {
        public int    Id                      { get; set; }
        public string StudentId               { get; set; }
        public int    AttemptId               { get; set; }
        public int    CheatingReportId        { get; set; }
        public double  SessionViolationRate   { get; set; }
        public decimal OverallSessionRiskScore { get; set; }
        public List<RiskQuestionResultDto> Questions { get; set; } = [];
    }
}
