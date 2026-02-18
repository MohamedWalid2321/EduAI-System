namespace PresentationLayer.Controllers
{
	public class AuthunticationController(IServiceManager serviceManager) : ApiControllerBase
	{
		private readonly IServiceManager _serviceManager = serviceManager;

		[HttpPost("login")]
		public async Task<IActionResult> LoginAsync(LoginRequest request)
		{
			var authResponse = await _serviceManager.AuthunticationService.GetTokenAsync(request.Email, request.Password);
			return Ok(authResponse);
		}
		[HttpPost("register")]
		public async Task<IActionResult> RegisterAsync([FromForm] RegisterRequest request,IFormFile? file)
		{
			await _serviceManager.AuthunticationService.RegisterAsync(request, file);
			return Ok();
		}
		[HttpPost("refresh")]
		public async Task<IActionResult> RefreshAsync(RefreshTokenRequest request)
		{
			var authResponse = await _serviceManager.AuthunticationService.GetRefreshTokenAsync(request.token, request.refreshToken);
			return Ok(authResponse);
		}
		[HttpPost("revoke_refresh_token")]
		public async Task<IActionResult> RevokeRefreshTokenAsync(RefreshTokenRequest request)
		{
			await _serviceManager.AuthunticationService.RevokeRefreshTokenAsync(request.token,request.refreshToken);
			return Ok() ;
		}

	}
}
