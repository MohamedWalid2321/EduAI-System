using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AttemptQuiz.Request
{
    public class AnswerRequestDto
    {
        public int QuestionId { get; set; }
        public int ChoiceId { get; set; }
    }
}
