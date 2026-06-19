using DomainLayer.Models;

namespace ServiceLayer.Specifications.CheatingReportSpecifications
{
    /// <summary>
    /// Fetches the RiskAssessmentResult for a given CheatingReport ID,
    /// eagerly loading all per-question results.
    /// </summary>
    public class RiskAssessmentResultByCheatingReportSpecification : BaseSpecification<RiskAssessmentResult, int>
    {
        public RiskAssessmentResultByCheatingReportSpecification(int cheatingReportId)
            : base(r => r.CheatingReportId == cheatingReportId)
        {
            AddInclude_2(query => query
                .Include(r => r.Questions));
        }
    }
}
