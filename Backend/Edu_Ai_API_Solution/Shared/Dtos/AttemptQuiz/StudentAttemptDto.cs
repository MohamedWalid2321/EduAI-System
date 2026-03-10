using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AttemptQuiz
{
    public class StudentAttemptDto
    {
        public string StudentId { get; set; }
        public int Score { get; set; }
        public DateTime SubmittedAt { get; set; }
    }
}
