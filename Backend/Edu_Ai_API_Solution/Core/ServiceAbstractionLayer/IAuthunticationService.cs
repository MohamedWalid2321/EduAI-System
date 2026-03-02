


namespace ServiceAbstractionLayer
{
	public interface IAuthunticationService
	{
		Task<AuthResponse> GetTokenAsync(string email, string password);
		Task RegisterAsync(RegisterRequest request, IFormFile? file);
		Task<AuthResponse> GetRefreshTokenAsync(string Token, string RefreshToken);
		Task RevokeRefreshTokenAsync(string Token, string RefreshToken);
		Task ConfirmEmailAsync(ConfirmEmailRequest request);
		Task ResendConfirmEmailAsync(ResendConfirmEmailRequest request);
		Task SendResetPasswordCodeAsync(string email);
		Task ResetPasswordAsync(ResetPasswordRequest request);

	}
}
