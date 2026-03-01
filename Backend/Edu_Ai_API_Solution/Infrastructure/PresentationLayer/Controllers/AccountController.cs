namespace PresentationLayer.Controllers
{
	[Route("me")]
	[ApiController]
	[Authorize]
	public class AccountController(IServiceManager serviceManager) : ControllerBase
	{
		private readonly IServiceManager _serviceManager = serviceManager;
		[HttpGet("")]
		public async Task<IActionResult> Info()
		{
			var user = await _serviceManager.UserService.GetUserProfileAsync(User.GetUserId()!);
			return Ok(user);
		}
		[HttpPut("info")]
		public async Task<IActionResult> Info([FromForm]  UpdateUserProfileRequest request, IFormFile file)
		{
			await _serviceManager.UserService.UpdateUserProfileAsync(User.GetUserId()!, request,file);
			return NoContent();
		}
		[HttpPut("change-password")]
		public async Task<IActionResult> ChangePassword([FromBody] ChangePasswordRequest request)
		{
			await _serviceManager.UserService.ChangePasswordAsync(User.GetUserId()!, request);

			return  NoContent() ;
		}
		[HttpPut("LevelUp/{id}")]
		public async Task<IActionResult> LevelUp([FromRoute]string id)
		{
			await _serviceManager.UserService.LevelUp(id);
			return NoContent();
		}

	}
}
