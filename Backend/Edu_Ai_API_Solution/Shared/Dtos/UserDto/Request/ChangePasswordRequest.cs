using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.UserDto.Request
{
	public class ChangePasswordRequest
	{
		public string CurrentPassword { get; set; }= null!;
		public string NewPassword { get; set; } = null!;
	}
}
