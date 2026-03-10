using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AttemptQuiz.Response
{
    public class StartQuizResponseDto
    {
        public int AttemptId { get; set; }
        public string Title { get; set; }
        public TimeSpan Duration { get; set; }
        public List<QuestionForStudentDto> Questions { get; set; }
    }
}
