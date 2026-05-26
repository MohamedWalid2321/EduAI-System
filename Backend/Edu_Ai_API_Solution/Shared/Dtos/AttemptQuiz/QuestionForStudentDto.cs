using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AttemptQuiz
{
    public class QuestionForStudentDto
    {
        public int Id { get; set; }
        public string QuestionText { get; set; }
        public bool IsAllowableToLookDown { get; set; }
        public List<ChoiceDto> Choices { get; set; }
    }
}
