using System.Text.Json;

namespace persistenceLayer.Auth
{
	public class JwtProvider(IOptions<JwtOptions> options) : IJwtProvider
	{
		private readonly JwtOptions _jwtOptions = options.Value;
		public (string Token, int ExpireIn) GenerateToken(ApplicationUser user, IEnumerable<String> Roles, IEnumerable<string> Permissions)
		{
			Claim[] claims = [
				new Claim(JwtRegisteredClaimNames.Sub, user.Id),
				new Claim(JwtRegisteredClaimNames.Email, user.Email!),
				new Claim(JwtRegisteredClaimNames.GivenName,user.FirstName !),
				new Claim(JwtRegisteredClaimNames.FamilyName, user.LastName !),
				new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString()),
				new Claim(nameof(user.DepartmentId),user.DepartmentId.ToString() ?? "Not Enrolled" ),
				new Claim(nameof(Roles),JsonSerializer.Serialize(Roles),JsonClaimValueTypes.JsonArray),
				new Claim(nameof(Permissions),JsonSerializer.Serialize(Permissions),JsonClaimValueTypes.JsonArray)
			];
			var SymmtricSecurityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtOptions.Key));
			var signingCredentials = new SigningCredentials(SymmtricSecurityKey, SecurityAlgorithms.HmacSha256);
			var expireIn = _jwtOptions.ExpiredSeconds; // 30 minutes
			var jwtToken = new JwtSecurityToken(
				issuer: _jwtOptions.Issuer,
				audience: _jwtOptions.Audience,
				claims: claims,
				notBefore: DateTime.UtcNow,
				expires: DateTime.UtcNow.AddSeconds(expireIn),
				signingCredentials: signingCredentials
				);
			return (new JwtSecurityTokenHandler().WriteToken(jwtToken), expireIn);
		}

		public string? ValidateToken(string token)
		{
			var tokenHandler = new JwtSecurityTokenHandler();
			var SymmtricSecurityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtOptions.Key));

			try
			{
				tokenHandler.ValidateToken(token, new TokenValidationParameters
				{
					IssuerSigningKey = SymmtricSecurityKey,
					ValidateIssuerSigningKey = true,
					ValidateIssuer = false,
					ValidateAudience = false,
					ClockSkew = TimeSpan.Zero
				}, out SecurityToken validatedToken);
				var jwtToken = (JwtSecurityToken)validatedToken;
				var userId = jwtToken.Claims.First(claim => claim.Type == JwtRegisteredClaimNames.Sub).Value;
				return userId;
			}
			catch
			{
				return null;
			}
		}
	}
}
