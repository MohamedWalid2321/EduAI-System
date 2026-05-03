namespace DomainLayer.Models
{
	public class Notification : BaseEntity<int>
	{
		public string Title { get; set; } = null!;
		public string Message { get; set; } = null!;
		public bool IsRead { get; set; } = false;
		public int NotificationBoxId { get; set; }
		public NotificationBox NotificationBox { get; set; } = null!;
	}
}