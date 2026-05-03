namespace ServiceLayer.Specifications.NotificationSpecifications
{
	public class NotificationByIdSpecification : BaseSpecification<Notification, int>
	{
		public NotificationByIdSpecification(int notificationId)
			: base(n => n.Id == notificationId)
		{
			AddInclude(n => n.NotificationBox);
		}
	}
}