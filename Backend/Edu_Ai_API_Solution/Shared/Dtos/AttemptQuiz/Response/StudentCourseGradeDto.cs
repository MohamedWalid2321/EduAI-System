using System;

namespace Shared.Dtos.AttemptQuiz.Response
{
    /// <summary>
    /// Represents a student's grade for a single quiz within a course.
    /// </summary>
    public class StudentCourseGradeDto
    {
        public int    AttemptId    { get; set; }
        public string QuizTitle    { get; set; } = string.Empty;
        public string QuizCode     { get; set; } = string.Empty;
        public int    Score        { get; set; }
        public double TotalMarks   { get; set; }
        public DateTime SubmittedAt { get; set; }
    }
}
