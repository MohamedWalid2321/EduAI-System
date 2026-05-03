using Shared.Dtos.NotificationDto;

namespace ServiceAbstractionLayer
{
	public interface INotificationService
	{
		Task<NotificationBoxResponse> GetUserNotificationsAsync(string userId, CancellationToken cancellationToken = default);
		Task MarkAsReadAsync(int notificationId, string userId, CancellationToken cancellationToken = default);
		Task MarkAllAsReadAsync(string userId, CancellationToken cancellationToken = default);
		Task CreateNotificationAsync(string userId, string title, string message, CancellationToken cancellationToken = default);
		Task DeleteNotificationAsync(int notificationId, string userId, CancellationToken cancellationToken = default);
	}
}