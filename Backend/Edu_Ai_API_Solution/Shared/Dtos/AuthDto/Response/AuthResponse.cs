using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AuthDto.Response
{
	public class AuthResponse
	{
		public string id { get; set; } = string.Empty;
		public string? Email { get; set; }
		public string FirstName { get; set; }= string.Empty;
		public string LastName { get; set; }= string.Empty;
		public string? ProfilePictureUrl { get; set; }
		public string token { get; set; } = string.Empty;
		public int ExpinresIn { get; set; }
		public string RefreshToken { get; set; } = string.Empty;
		public DateTime RefreshTokenExpiration { get; set; }
	}
}
