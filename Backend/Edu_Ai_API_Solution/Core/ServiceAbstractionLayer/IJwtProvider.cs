namespace ServiceAbstractionLayer
{
	public interface IJwtProvider
	{
		(string Token, int ExpireIn) GenerateToken(ApplicationUser user);
		string? ValidateToken(string token);
	}
}
