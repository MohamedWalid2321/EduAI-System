namespace Shared.Dtos.NotificationDto
{
	public class NotificationBoxResponse
	{
		public int Id { get; set; }
		public int UnreadCount { get; set; }
		public IEnumerable<NotificationResponse> Notifications { get; set; } = [];
	}
}