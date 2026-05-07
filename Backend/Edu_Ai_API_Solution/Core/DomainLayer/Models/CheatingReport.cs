namespace DomainLayer.Models
{
    public class CheatingReport : BaseEntity<int>
    {
        public int QuizAttemptId { get; set; }
        public QuizAttempt QuizAttempt { get; set; }

        public string StudentId { get; set; }
        public ApplicationUser Student { get; set; }

        public ICollection<CheatingViolation> Violations { get; set; } = [];
    }
}