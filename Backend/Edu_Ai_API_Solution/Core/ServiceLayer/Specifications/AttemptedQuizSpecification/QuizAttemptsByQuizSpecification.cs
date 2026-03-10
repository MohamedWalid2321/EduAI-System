using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    public class QuizAttemptsByQuizSpecification: BaseSpecification<QuizAttempt,int>
    {
        public QuizAttemptsByQuizSpecification(string quizCode)
        : base(q => q.QuizCode == quizCode && q.IsSubmitted) 
        {
            
            AddInclude_2(query => query.Include(q => q.User));
        }
    }
}
