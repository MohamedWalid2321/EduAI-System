using Microsoft.AspNetCore.Identity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public sealed class ApplicationUser:IdentityUser
	{
		public string? FirstName { get; set; } = string.Empty;
		public string? LastName { get; set; }= string.Empty;
		public string? ProfilePictureUrl { get; set; } = string.Empty;
		public string? ProfilePictureBase64 { get; set; } = string.Empty;
		public DateOnly DateOfBirth { get; set; }
		public bool IsDisabled { get; set; }
		public List<RefreshToken> RefreshTokens { get; set; } = [];

	}
}
