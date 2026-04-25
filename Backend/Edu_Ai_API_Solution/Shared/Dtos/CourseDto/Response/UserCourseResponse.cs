namespace Shared.Dtos.CourseDto.Response
{
	public class UserCourseResponse
	{
		public int Id { get; set; }
		public string UserId { get; set; } = string.Empty;
		public string UserName { get; set; } = string.Empty;
		public string UserEmail { get; set; } = string.Empty;
		public int CourseId { get; set; }
		public string CourseTitle { get; set; } = string.Empty;
		public DateTime EnrolledAt { get; set; }
		public string? EnrolledBy { get; set; }
	}
}
