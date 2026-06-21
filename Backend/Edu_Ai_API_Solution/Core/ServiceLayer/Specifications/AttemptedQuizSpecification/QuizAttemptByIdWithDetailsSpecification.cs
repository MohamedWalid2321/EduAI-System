using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    /// <summary>
    /// Fetches a single submitted attempt by its ID,
    /// eagerly loading the student, their answers, the answered question text,
    /// and the chosen / correct choice texts.
    /// </summary>
    public class QuizAttemptByIdWithDetailsSpecification : BaseSpecification<QuizAttempt, int>
    {
        public QuizAttemptByIdWithDetailsSpecification(int attemptId)
            : base(a => a.Id == attemptId)
        {
            // Load the quiz (to access TotalMarks)
            AddInclude_2(query => query
                .Include(a => a.Quiz));

            // Load the student user
            AddInclude_2(query => query
                .Include(a => a.User));

            // Load each student answer → question text + all choices (to get correct one)
            AddInclude_2(query => query
                .Include(a => a.StudentAnswers)
                    .ThenInclude(sa => sa.QuizQuestion)
                        .ThenInclude(qq => qq.QuestionChoices));

            // Load each student answer → the choice the student selected
            AddInclude_2(query => query
                .Include(a => a.StudentAnswers)
                    .ThenInclude(sa => sa.QuestionChoice));
        }
    }
}
