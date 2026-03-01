using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.UserDto.Response
{
	public class UserResponse
	{
		public string Id { get; set; }= string.Empty;
		public string FirstName { get; set; } = string.Empty;
		public string LastName { get; set; } = string.Empty;
		public string Email { get; set; } = string.Empty;
		public bool IsDisabled { get; set; }
		public string AcademicYear { get; set; } = string.Empty;
		public int DepartmentId { get; set; }
		public IEnumerable<string> Roles { get; set; } = [];
	}
}
