namespace Shared.Dtos.NotificationDto
{
	public class NotificationResponse
	{
		public int Id { get; set; }
		public string Title { get; set; } = null!;
		public string Message { get; set; } = null!;
		public bool IsRead { get; set; }
		public DateTime? CreatedAt { get; set; }
	}
}