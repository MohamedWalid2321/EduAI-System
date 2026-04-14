using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.QuizDto.Response
{
    public class QuestionChoiceResponseDto
    {
        public int Id { get; set; }

        public string ChoiceText { get; set; }

        public bool IsCorrect { get; set; }
    }
}
