namespace PresentationLayer.Controllers
{
	[Route("me")]
	[ApiController]
	[Authorize]
	public class AccountController(IServiceManager serviceManager) : ControllerBase
	{
		private readonly IServiceManager _serviceManager = serviceManager;

		[HttpGet("")]
		public async Task<IActionResult> Info(CancellationToken cancellationToken)
		{
			var user = await _serviceManager.UserService.GetUserProfileAsync(User.GetUserId()!, cancellationToken);
			return Ok(user);
		}

		[HttpPut("info")]
		public async Task<IActionResult> Info([FromForm]  UpdateUserProfileRequest request, IFormFile file, CancellationToken cancellationToken)
		{
			await _serviceManager.UserService.UpdateUserProfileAsync(User.GetUserId()!, request, file, cancellationToken);
			return Ok();
		}
		[HttpPut("change-password")]
		public async Task<IActionResult> ChangePassword([FromBody] ChangePasswordRequest request, CancellationToken cancellationToken)
		{
			await _serviceManager.UserService.ChangePasswordAsync(User.GetUserId()!, request, cancellationToken);

			return  Ok() ;
		}
		[HttpPut("LevelUp")]
		[HasPermission(Permissions.LevelUp)]
		public async Task<IActionResult> LevelUp(CancellationToken cancellationToken)
		{
			await _serviceManager.UserService.LevelUp(User.GetUserId()!, cancellationToken);
			return Ok();
		}
	}
}
