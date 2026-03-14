using DomainLayer.Enums;
using DomainLayer.Models;

namespace ServiceLayer.Specifications.UserCourseSpecifications
{
	public class UserEnrolledCoursesSpecification : BaseSpecification<Course, int>
	{
		public UserEnrolledCoursesSpecification(string userId)
			: base(c => c.UserCourses.Any(uc => uc.UserId == userId && uc.Status == EnrollmentStatus.Active))
		{
			AddInclude(c => c.Assessments);
			AddInclude(c => c.Departments);
		}
	}
}
