namespace PresentationLayer.Controllers
{
	public class AuthunticationController(IServiceManager serviceManager) : ApiControllerBase
	{
		private readonly IServiceManager _serviceManager = serviceManager;

		[HttpPost("login")]
		public async Task<IActionResult> LoginAsync(LoginRequest request, CancellationToken cancellationToken)
		{
			var authResponse = await _serviceManager.AuthunticationService.GetTokenAsync(request.Email, request.Password, cancellationToken);
			return Ok(authResponse);
		}

		[HttpPost("register")]
		public async Task<IActionResult> RegisterAsync([FromForm] RegisterRequest request, IFormFile? file, CancellationToken cancellationToken)
		{
			await _serviceManager.AuthunticationService.RegisterAsync(request, file, cancellationToken);
			return Ok();
		}

		[HttpPost("refresh")]
		public async Task<IActionResult> RefreshAsync(RefreshTokenRequest request, CancellationToken cancellationToken)
		{
			var authResponse = await _serviceManager.AuthunticationService.GetRefreshTokenAsync(request.token, request.refreshToken, cancellationToken);
			return Ok(authResponse);
		}

		[HttpPost("revoke_refresh_token")]
		public async Task<IActionResult> RevokeRefreshTokenAsync(RefreshTokenRequest request, CancellationToken cancellationToken)
		{
			await _serviceManager.AuthunticationService.RevokeRefreshTokenAsync(request.token, request.refreshToken, cancellationToken);
			return Ok();
		}

		[HttpPost("confirm_email")]
		public async Task<IActionResult> ConfirmEmailAync(ConfirmEmailRequest request, CancellationToken cancellationToken)
		{
			
			await _serviceManager.AuthunticationService.ConfirmEmailAsync(request, cancellationToken);
			return Ok();
		}
		[HttpPost("resend_confirm_email")]
		public async Task<IActionResult> ResendConfirmEmailAync(ResendConfirmEmailRequest request, CancellationToken cancellationToken)
		{
			
			await _serviceManager.AuthunticationService.ResendConfirmEmailAsync(request, cancellationToken);
			return Ok();
		}
		[HttpPost("forget-password")]
		public async Task<IActionResult> ForgetPassword([FromBody] ForgetPasswordRequest request, CancellationToken cancellationToken)
		{
			await _serviceManager.AuthunticationService.SendResetPasswordCodeAsync(request.Email, cancellationToken);
			return Ok();
		}
		[HttpPost("reset-password")]
		public async Task<IActionResult> ResetPassword([FromBody] ResetPasswordRequest request, CancellationToken cancellationToken)
		{
			await _serviceManager.AuthunticationService.ResetPasswordAsync(request, cancellationToken);
			return Ok();
		}

	}
}
