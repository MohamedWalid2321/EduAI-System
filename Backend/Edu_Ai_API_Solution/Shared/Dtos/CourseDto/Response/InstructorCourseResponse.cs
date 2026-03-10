namespace Shared.Dtos.CourseDto.Response
{
	public class InstructorCourseResponse
	{
		public int Id { get; set; }
		public string InstructorId { get; set; } = string.Empty;
		public string InstructorName { get; set; } = string.Empty;
		public string InstructorEmail { get; set; } = string.Empty;
		public int CourseId { get; set; }
		public string CourseTitle { get; set; } = string.Empty;
		public DateTime AssignedAt { get; set; }
	}
}