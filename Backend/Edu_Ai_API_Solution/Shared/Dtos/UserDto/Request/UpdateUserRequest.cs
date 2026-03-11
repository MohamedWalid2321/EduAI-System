using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.UserDto.Request
{
	public class UpdateUserRequest
	{
		public string FirstName { get; set; } = string.Empty;
		public string LastName { get; set; } = string.Empty;
		public string Email { get; set; } = string.Empty;
		public string AcademicYear { get; set; } = string.Empty;	
		public int? DepartmentId { get; set; }
		public IList<string> Roles { get; set; } = [];
	}
}
