using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.QuestionSpecifications
{
    public class QuestionInQuizSpecifications : BaseSpecification<QuizQuestion, int>
    {
        public QuestionInQuizSpecifications(int? quizId, string? questionContent)
            : base(p => (string.IsNullOrEmpty(questionContent) || p.QuestionText.Contains(questionContent))
                        &&
                        (!quizId.HasValue || p.QuizId == quizId.Value)
                  )
        {

            AddInclude_2(query => query
                        .Include(q => q.QuestionChoices));
        }
        public QuestionInQuizSpecifications(int? quizId)
            : base(p => (!quizId.HasValue || p.QuizId == quizId.Value))
        {
            AddInclude_2(query => query
                        .Include(q => q.QuestionChoices));
        }
        public QuestionInQuizSpecifications(int? quizId, int? questionId)
            : base(p => ((!quizId.HasValue || p.QuizId == quizId.Value))
                        &&
                        (!questionId.HasValue || p.Id == questionId.Value))
        {
            AddInclude_2(query => query
                        .Include(q => q.QuestionChoices));

        }
    }
}

