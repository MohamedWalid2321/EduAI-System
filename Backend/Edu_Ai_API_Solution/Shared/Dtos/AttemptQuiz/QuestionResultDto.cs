using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AttemptQuiz
{
    public class QuestionResultDto
    {
        public string QuestionText { get; set; }
        public string StudentChoice { get; set; }
        public string CorrectChoice { get; set; }
        public bool IsCorrect { get; set; }
    }
}
