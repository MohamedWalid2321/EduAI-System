using System;

namespace Shared.Dtos.LectureDto.Response
{
	public class LectureResponse
	{
		public int Id { get; set; }
		public string Title { get; set; } = string.Empty;
		public string Description { get; set; } = string.Empty;
		public DateTime ScheduledAt { get; set; }
		public bool IsActive { get; set; }
		public string CreatedByName { get; set; } = string.Empty;
		public int CourseId { get; set; }
	}
}