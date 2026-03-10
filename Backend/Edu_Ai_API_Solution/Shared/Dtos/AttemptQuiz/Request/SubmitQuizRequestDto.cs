using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AttemptQuiz.Request
{
    public class SubmitQuizRequestDto
    {
        public List<AnswerRequestDto> Answers { get; set; }
    }
}
