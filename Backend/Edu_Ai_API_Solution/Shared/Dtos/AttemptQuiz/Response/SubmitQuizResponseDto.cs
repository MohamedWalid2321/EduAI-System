using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AttemptQuiz.Response
{
    public class SubmitQuizResponseDto
    {
        public string QuizCode { get; set; }
        public string QuizTitle { get; set; }
        public int Score { get; set; }
        public int TotalQuestions { get; set; }
        public double Percentage => TotalQuestions == 0 ? 0 : (double)Score / TotalQuestions * 100;
        public List<QuestionResultDto> Questions { get; set; }
    }
}
