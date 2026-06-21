using System;
using System.Collections.Generic;

namespace Shared.Dtos.AttemptQuiz.Response
{
    /// <summary>
    /// Represents a single student answer within a quiz attempt detail response.
    /// </summary>
    public class AttemptAnswerDto
    {
        public string QuestionText { get; set; }
        public string StudentChoice { get; set; }
        public string CorrectChoice { get; set; }
        public bool IsCorrect { get; set; }
    }

    /// <summary>
    /// Full details of one quiz attempt (one student's submission for a quiz).
    /// </summary>
    public class QuizAttemptDetailsDto
    {
        public int AttemptId { get; set; }
        public string StudentFullName { get; set; }
        public string StudentId { get; set; }
        public int Score { get; set; }
        public double QuizTotalMarks { get; set; }
        public DateTime SubmittedAt { get; set; }
        public List<AttemptAnswerDto> StudentAnswers { get; set; }
    }
}
