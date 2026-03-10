using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    public class ExistingStudentAnswersSpecification: BaseSpecification<StudentAnswer,int>
    {
        public ExistingStudentAnswersSpecification(int attemptId) : base(q => q.QuizAttemptId == attemptId)
        {
            AddInclude_2(query => query
                        .Include(q => q.QuizAttempt)
                        .ThenInclude(qa => qa.Quiz)
                        .ThenInclude(q => q.QuizQuestions));
        }
    }
}
