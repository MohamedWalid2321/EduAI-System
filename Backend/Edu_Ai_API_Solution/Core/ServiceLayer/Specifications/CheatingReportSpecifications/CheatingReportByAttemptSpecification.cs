using DomainLayer.Models;

namespace ServiceLayer.Specifications.CheatingReportSpecifications
{
    public class CheatingReportByAttemptSpecification : BaseSpecification<CheatingReport, int>
    {
        public CheatingReportByAttemptSpecification(int attemptId)
            : base(r => r.QuizAttemptId == attemptId)
        {
            AddInclude(r => r.Violations);
            AddInclude(r => r.Student);
        }
    }
}