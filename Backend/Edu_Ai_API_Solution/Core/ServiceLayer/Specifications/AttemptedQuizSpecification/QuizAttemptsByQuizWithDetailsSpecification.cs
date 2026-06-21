using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AttemptedQuizSpecification
{
    /// <summary>
    /// Fetches all submitted attempts for a specific quiz (by quizId),
    /// eagerly loading the student, their answers, the answered question text,
    /// and the chosen / correct choice texts.
    /// </summary>
    public class QuizAttemptsByQuizWithDetailsSpecification : BaseSpecification<QuizAttempt, int>
    {
        public QuizAttemptsByQuizWithDetailsSpecification(int quizId)
            : base(q => q.QuizId == quizId && q.IsSubmitted)
        {
            // Load the quiz (to access TotalMarks)
            AddInclude_2(query => query
                .Include(a => a.Quiz));

            // Load the student user
            AddInclude_2(query => query
                .Include(a => a.User));

            // Load each student answer → question text + all choices
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
