


namespace ServiceAbstractionLayer
{
	public interface IAuthunticationService
	{
		Task<AuthResponse> GetTokenAsync(string email, string password);
		Task RegisterAsync(RegisterRequest request, IFormFile? file);
		Task<AuthResponse> GetRefreshTokenAsync(string Token, string RefreshToken);
		Task RevokeRefreshTokenAsync(string Token, string RefreshToken);
	}
}
