using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.QuizSpecifications
{
	public class QuizSpecification : BaseSpecification<Quiz, int>
	{
		public QuizSpecification(int id) : base(q => q.Id == id)
		{
            
            AddInclude_2(query => query
                        .Include(q => q.QuizQuestions)
                        .ThenInclude(qq => qq.QuestionChoices));

        }

		public QuizSpecification() : base(null)
		{
            AddInclude_2(query => query
                        .Include(q => q.QuizQuestions)
                        .ThenInclude(qq => qq.QuestionChoices));
        }
	}
}