namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    public class AttemptWithUserSpecification : BaseSpecification<QuizAttempt, int>
    {
        public AttemptWithUserSpecification(int attemptId)
            : base(q => q.Id == attemptId)
        {
            AddInclude(q => q.User);
        }
    }
}