using Shared.Dtos.NotificationDto;

namespace PresentationLayer.Controllers
{
	[Authorize]
	public class NotificationController(IServiceManager serviceManager) : ApiControllerBase
	{
		private readonly IServiceManager _serviceManager = serviceManager;
		[HttpGet]
		public async Task<ActionResult<NotificationBoxResponse>> GetMyNotifications(CancellationToken cancellationToken)
		{
			var userId = User.GetUserId();
			var result = await _serviceManager.NotificationService.GetUserNotificationsAsync(userId, cancellationToken);
			return Ok(result);
		}
		[HttpPut("{notificationId:int}/read")]
		public async Task<IActionResult> MarkAsRead(int notificationId, CancellationToken cancellationToken)
		{
			var userId = User.GetUserId();
			await _serviceManager.NotificationService.MarkAsReadAsync(notificationId, userId, cancellationToken);
			return NoContent();
		}
		[HttpPut("read-all")]
		public async Task<IActionResult> MarkAllAsRead(CancellationToken cancellationToken)
		{
			var userId = User.GetUserId();
			await _serviceManager.NotificationService.MarkAllAsReadAsync(userId, cancellationToken);
			return NoContent();
		}
		[HttpDelete("{notificationId:int}")]
		public async Task<IActionResult> DeleteNotification(int notificationId, CancellationToken cancellationToken)
		{
			var userId = User.GetUserId();
			await _serviceManager.NotificationService.DeleteNotificationAsync(notificationId, userId, cancellationToken);
			return NoContent();
		}
	}
}