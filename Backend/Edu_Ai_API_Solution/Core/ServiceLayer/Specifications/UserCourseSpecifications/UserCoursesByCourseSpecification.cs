using DomainLayer.Models;

namespace ServiceLayer.Specifications.UserCourseSpecifications
{
	public class UserCoursesByCourseSpecification : BaseSpecification<UserCourse, int>
	{
		public UserCoursesByCourseSpecification(int courseId)
			: base(uc => uc.CourseId == courseId)
		{
			AddInclude(uc => uc.User);
			AddInclude(uc => uc.Course);
		}
	}
}
