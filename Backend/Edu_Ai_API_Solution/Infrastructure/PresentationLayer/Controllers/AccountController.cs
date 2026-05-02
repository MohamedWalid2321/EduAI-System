namespace PresentationLayer.Controllers
{
	[Route("me")]
	[ApiController]
	[Authorize]
	public class AccountController(IServiceManager serviceManager, ICacheService cacheService) : ControllerBase
	{
		private readonly IServiceManager _serviceManager = serviceManager;

		[HttpGet("")]
		[Cache(120)]
		public async Task<IActionResult> Info(CancellationToken cancellationToken)
		{
			var user = await _serviceManager.UserService.GetUserProfileAsync(User.GetUserId()!, cancellationToken);
			return Ok(user);
		}

		[HttpPut("info")]
		public async Task<IActionResult> UpdateInfo([FromForm] UpdateUserProfileRequest request, IFormFile file, CancellationToken cancellationToken)
		{
			var userId = User.GetUserId()!;
			await _serviceManager.UserService.UpdateUserProfileAsync(userId, request, file, cancellationToken);
			await cacheService.RemoveAsync($"/me|user:{userId}");
			await cacheService.RemoveByPatternAsync($"/api/course*|user:{userId}");
			// Admin user list reflects updated profile data
			await cacheService.RemoveByPatternAsync(UsersController.UsersPattern);
			return Ok();
		}

		[HttpPut("change-password")]
		public async Task<IActionResult> ChangePassword([FromBody] ChangePasswordRequest request, CancellationToken cancellationToken)
		{
			var userId = User.GetUserId()!;
			await _serviceManager.UserService.ChangePasswordAsync(userId, request, cancellationToken);
			await cacheService.RemoveAsync($"/me|user:{userId}");
			return Ok();
		}

		[HttpPut("LevelUp")]
		[HasPermission(Permissions.LevelUp)]
		public async Task<IActionResult> LevelUp(CancellationToken cancellationToken)
		{
			var userId = User.GetUserId()!;
			await _serviceManager.UserService.LevelUp(userId, cancellationToken);
			await cacheService.RemoveAsync($"/me|user:{userId}");
			await cacheService.RemoveByPatternAsync($"/api/course*|user:{userId}");
			// Level up changes academic year — visible in admin user list
			await cacheService.RemoveByPatternAsync(UsersController.UsersPattern);
			return Ok();
		}
	}
}
