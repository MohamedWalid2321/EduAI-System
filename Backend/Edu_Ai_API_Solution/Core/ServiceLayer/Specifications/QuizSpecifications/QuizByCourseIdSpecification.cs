using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.QuizSpecifications
{
	public class QuizByCourseIdSpecification : BaseSpecification<Quiz, int>
	{
		// For instructors / admins — returns all quizzes regardless of IsActive
		public QuizByCourseIdSpecification(int courseId) : base(q => q.CourseId == courseId)
		{
            AddInclude_2(query => query
                        .Include(q => q.QuizQuestions)
                        .ThenInclude(qq => qq.QuestionChoices));
        }

		// For students — returns only active quizzes
		public QuizByCourseIdSpecification(int courseId, bool onlyActive) 
			: base(q => q.CourseId == courseId && (!onlyActive || q.IsActive))
		{
            AddInclude_2(query => query
                        .Include(q => q.QuizQuestions)
                        .ThenInclude(qq => qq.QuestionChoices));
        }
	}
}