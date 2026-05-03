using DomainLayer.Models;

namespace ServiceLayer.Specifications.NotificationSpecifications
{
	public class NotificationBoxByUserSpecification : BaseSpecification<NotificationBox, int>
	{
		public NotificationBoxByUserSpecification(string userId)
			: base(nb => nb.UserId == userId)
		{
			AddInclude(nb => nb.Notifications);
		}
	}
}