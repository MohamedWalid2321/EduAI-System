
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
    public class QuizAttempt:BaseEntity<int>
    {
        public DateTime SubmittedAt { get; set; } // when the quiz attempt was submitted
        public bool IsSubmitted { get; set; } = false; // whether the quiz attempt has been submitted or is still in progress
        public int Score { get; set; } // the score obtained in the quiz attempt
        public string QuizCode { get; set; } // the code of the quiz being attempted
        public int QuizId { get; set; } // foreign key to the Quiz entity
        public Quiz Quiz { get; set; } // navigation property to the Quiz entity
        public string StudentId { get; set; } // foreign key to the ApplicationUser entity
        public ApplicationUser User { get; set; } // navigation property to the ApplicationUser entity
        public ICollection<StudentAnswer> StudentAnswers { get; set; }
    }
}
