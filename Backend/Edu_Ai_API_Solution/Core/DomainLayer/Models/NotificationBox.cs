namespace DomainLayer.Models
{
	public class NotificationBox : BaseEntity<int>
	{
		public string UserId { get; set; } = null!;
		public ApplicationUser User { get; set; } = null!;
		public ICollection<Notification> Notifications { get; set; } = [];
	}
}