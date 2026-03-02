using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.UserDto.Response
{
	public class UserProfileResponse
	{
		public string Email { get; set; } = null!;
		public string FirstName { get; set; } = null!;
		public string LastName { get; set; } = null!;
		public string? ProfilePictureUrl { get; set; }
		public DateOnly DateOfBirth { get; set; }
	}
}
