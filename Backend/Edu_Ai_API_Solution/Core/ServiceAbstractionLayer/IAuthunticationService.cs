namespace ServiceAbstractionLayer
{
	public interface IAuthunticationService
	{
		Task<AuthResponse> GetTokenAsync(string email, string password, CancellationToken cancellationToken = default);
		Task RegisterAsync(RegisterRequest request, IFormFile? file, CancellationToken cancellationToken = default);
		Task<AuthResponse> GetRefreshTokenAsync(string Token, string RefreshToken, CancellationToken cancellationToken = default);
		Task RevokeRefreshTokenAsync(string Token, string RefreshToken, CancellationToken cancellationToken = default);
		Task ConfirmEmailAsync(ConfirmEmailRequest request, CancellationToken cancellationToken = default);
		Task ResendConfirmEmailAsync(ResendConfirmEmailRequest request, CancellationToken cancellationToken = default);
		Task SendResetPasswordCodeAsync(string email, CancellationToken cancellationToken = default);
		Task ResetPasswordAsync(ResetPasswordRequest request, CancellationToken cancellationToken = default);
	}
}
