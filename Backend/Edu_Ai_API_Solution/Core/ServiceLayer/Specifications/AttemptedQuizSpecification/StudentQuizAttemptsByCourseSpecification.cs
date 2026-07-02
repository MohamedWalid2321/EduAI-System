using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    /// <summary>
    /// Fetches all submitted quiz attempts for a specific student within a specific course,
    /// eagerly loading the related Quiz so title/code/totalMarks are available.
    /// </summary>
    public class StudentQuizAttemptsByCourseSpecification : BaseSpecification<QuizAttempt, int>
    {
        public StudentQuizAttemptsByCourseSpecification(int courseId, string studentId)
            : base(a => a.StudentId == studentId
                     && a.IsSubmitted
                     && a.Quiz.CourseId == courseId)
        {
            AddInclude(a => a.Quiz);
        }
    }
}
