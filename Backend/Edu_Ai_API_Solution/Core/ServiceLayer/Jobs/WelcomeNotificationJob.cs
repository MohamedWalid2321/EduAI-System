using ServiceAbstractionLayer;

namespace ServiceLayer.Jobs
{
	public class WelcomeNotificationJob(INotificationService notificationService)
	{
		private readonly INotificationService _notificationService = notificationService;

		public async Task SendWelcomeNotificationAsync(string userId, string firstName)
		{
			await _notificationService.CreateNotificationAsync(
				userId,
				"Welcome to Lumino! 🎉",
				$"Hi {firstName}, welcome to Lumino! We're excited to have you on board. Start exploring your courses and assignments.");
		}
	}
}