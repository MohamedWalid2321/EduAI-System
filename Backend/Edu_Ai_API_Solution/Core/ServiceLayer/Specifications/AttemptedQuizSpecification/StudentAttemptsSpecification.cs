using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    public class StudentAttemptsSpecification : BaseSpecification<QuizAttempt, int>
    {
        public StudentAttemptsSpecification(string studentId)
        : base(a => a.StudentId == studentId && a.IsSubmitted)
        {
            AddInclude(a => a.Quiz);
        }
    }
}
