using System.ComponentModel.DataAnnotations;

namespace Shared.Dtos.AttemptQuiz.Request
{
    public class UpdateAttemptScoreRequestDto
    {
        [Required]
        [Range(0, int.MaxValue, ErrorMessage = "Score must be a non-negative value.")]
        public int Score { get; set; }
    }
}
