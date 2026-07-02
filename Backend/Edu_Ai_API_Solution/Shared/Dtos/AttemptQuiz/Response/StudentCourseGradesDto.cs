using System;

namespace Shared.Dtos.AttemptQuiz.Response
{
    public class StudentCourseGradesDto
    {
        public int AttemptId { get; set; }
        public string QuizTitle { get; set; }
        public string QuizCode { get; set; }
        public int Score { get; set; }
        public double TotalMarks { get; set; }
        public DateTime SubmittedAt { get; set; }
    }
}
