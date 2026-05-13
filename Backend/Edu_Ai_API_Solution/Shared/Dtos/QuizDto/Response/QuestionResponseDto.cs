using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.QuizDto.Response
{
    public class QuestionResponseDto
    {
        public int Id { get; set; }
        public bool IsActive { get; set; }  
        public string QuestionText { get; set; }
        public string QuestionType { get; set; } // e.g. "MultipleChoice" or "TrueFalse"
        public bool IsAllowableToLookDown { get; set; }
        public List<QuestionChoiceResponseDto> QuestionChoices { get; set; }
    }
}
