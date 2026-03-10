using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    public class HasAttemptedQuizSpecification: BaseSpecification<QuizAttempt,int>
    {
        public HasAttemptedQuizSpecification(int quizId, string studentId) 
                                            : base(q => q.QuizId == quizId && q.StudentId == studentId)
        {
            AddInclude_2(query => query
                        .Include(q => q.Quiz)
                        .ThenInclude(qq => qq.QuizQuestions));

        }

    }
}
