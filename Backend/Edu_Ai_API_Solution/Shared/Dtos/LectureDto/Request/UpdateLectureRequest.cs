using System;

namespace Shared.Dtos.LectureDto.Request
{
	public class UpdateLectureRequest
	{
		public string Title { get; set; } = string.Empty;
		public string Description { get; set; } = string.Empty;
		public DateTime ScheduledAt { get; set; }
	}
}