using DomainLayer.Models;

namespace ServiceLayer.Specifications.CheatingReportSpecifications
{
    public class CheatingReportsByQuizSpecification : BaseSpecification<CheatingReport, int>
    {
        public CheatingReportsByQuizSpecification(int quizId)
            : base(r => r.QuizAttempt.QuizId == quizId)
        {
            AddInclude(r => r.Violations);
            AddInclude(r => r.Student);
        }
    }
}