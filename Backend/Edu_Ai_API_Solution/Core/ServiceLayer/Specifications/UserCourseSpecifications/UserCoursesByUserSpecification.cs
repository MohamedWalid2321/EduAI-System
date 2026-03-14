using DomainLayer.Models;

namespace ServiceLayer.Specifications.UserCourseSpecifications
{
	public class UserCoursesByUserSpecification : BaseSpecification<UserCourse, int>
	{
		public UserCoursesByUserSpecification(string userId)
			: base(uc => uc.UserId == userId)
		{
			AddInclude(uc => uc.Course);
		}
	}
}
