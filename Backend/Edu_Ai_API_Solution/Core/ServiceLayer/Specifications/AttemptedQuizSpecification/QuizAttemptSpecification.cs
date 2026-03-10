using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    public  class QuizAttemptSpecification : BaseSpecification<QuizAttempt,int>
    {
        public QuizAttemptSpecification(int attemptId , string studentId) 
                                            : base(q => q.Id == attemptId && q.StudentId == studentId)
        {
            AddInclude_2(query => query
                        .Include(q => q.Quiz)
                        .ThenInclude(qq => qq.QuizQuestions));
        }

    }
}
