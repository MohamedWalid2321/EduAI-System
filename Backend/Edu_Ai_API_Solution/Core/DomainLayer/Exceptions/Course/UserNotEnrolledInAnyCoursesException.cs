namespace DomainLayer.Exceptions.Course
{
	public class UserNotEnrolledInAnyCoursesException : BadRequestException
	{
		public UserNotEnrolledInAnyCoursesException(string userId)
			: base($"User with ID '{userId}' is not enrolled in any courses.")
		{
		}
	}
}