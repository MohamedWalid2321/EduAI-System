namespace Shared.Dtos.LectureDto.Response
{
	public class LectureJoinResponse
	{
		public int LectureId { get; set; }
		public string RoomName { get; set; } = string.Empty;
		public string JitsiDomain { get; set; } = string.Empty;     // e.g. "meet.jit.si"
		public string DisplayName { get; set; } = string.Empty;
		public string JitsiUrl { get; set; } = string.Empty;        // Full URL: https://{domain}/{roomName}
		public string ModeratorEmail { get; set; } = string.Empty;
	}
}