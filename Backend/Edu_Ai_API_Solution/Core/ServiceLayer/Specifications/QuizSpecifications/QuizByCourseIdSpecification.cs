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
		public QuizByCourseIdSpecification(int courseId) : base(q => q.CourseId == courseId)
		{
			AddInclude(q => q.QuizQuestions);
		}
	}
}