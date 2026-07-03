using System;

namespace Shared.Dtos.QuizDto.Response
{
    /// <summary>
    /// Returned to students when listing quizzes for a course.
    /// Includes the quiz's base information plus, when the student has already
    /// submitted an attempt, their score and the submission timestamp.
    /// </summary>
    public class QuizForStudentResponseDto : QuizResponseDto
    {
        /// <summary>
        /// True when the student has already submitted this quiz.
        /// </summary>
        public bool IsSubmitted { get; set; }

        /// <summary>
        /// The student's score. Null if the quiz has not been submitted yet.
        /// </summary>
        public int? Score { get; set; }

        /// <summary>
        /// When the student submitted the attempt. Null if not yet submitted.
        /// </summary>
        public DateTime? SubmittedAt { get; set; }
    }
}
