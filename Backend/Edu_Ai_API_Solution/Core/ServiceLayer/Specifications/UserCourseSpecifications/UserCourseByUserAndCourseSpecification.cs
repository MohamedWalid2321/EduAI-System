using DomainLayer.Models;

namespace ServiceLayer.Specifications.UserCourseSpecifications
{
	public class UserCourseByUserAndCourseSpecification : BaseSpecification<UserCourse, int>
	{
		public UserCourseByUserAndCourseSpecification(string userId, int courseId)
			: base(uc => uc.UserId == userId && uc.CourseId == courseId) { }
	}
}
