namespace ServiceAbstractionLayer
{
	public interface IJwtProvider
	{
		(string Token, int ExpireIn) GenerateToken(ApplicationUser user, IEnumerable<String> Roles, IEnumerable<string> Permissions);
		string? ValidateToken(string token);
	}
}
