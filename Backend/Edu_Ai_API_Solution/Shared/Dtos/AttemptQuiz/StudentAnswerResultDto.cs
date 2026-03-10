using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AttemptQuiz
{
    public class StudentAnswerResultDto
    {
        public int QuestionId { get; set; }
        public int SelectedChoiceId { get; set; }
        public bool IsCorrect { get; set; }
    }
}
