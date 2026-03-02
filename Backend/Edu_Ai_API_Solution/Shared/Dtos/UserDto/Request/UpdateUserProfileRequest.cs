using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.UserDto.Request
{
	public class UpdateUserProfileRequest
	{
		public string FirstName { get; set; } = null!;
		public string LastName { get; set; } = null!;
		public DateOnly DateOfBirth { get; set; }
		public string AcademicYear { get; set; } = null!;


	}
}
