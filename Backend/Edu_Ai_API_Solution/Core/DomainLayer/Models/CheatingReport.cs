namespace DomainLayer.Models
{
    public class CheatingReport : BaseEntity<int>
    {
        public int QuizAttemptId { get; set; }
        public QuizAttempt QuizAttempt { get; set; }

        public string StudentId { get; set; }
        public ApplicationUser Student { get; set; }

        public ICollection<CheatingViolation> Violations { get; set; } = [];

        /// <summary>
        /// Final normalized risk score (0–N) calculated by the background job
        /// after cohort min-max normalization. Null until the job completes.
        /// </summary>
        public decimal? RiskScore { get; set; }
    }
}