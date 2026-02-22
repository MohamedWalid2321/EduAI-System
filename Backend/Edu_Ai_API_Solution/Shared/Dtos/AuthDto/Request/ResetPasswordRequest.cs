using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AuthDto.Request
{
	public class ResetPasswordRequest
	{
		public string Email { get; set; }= null!;
		public string Code { get; set; }= null!;
		public string NewPassword { get; set; } = null!;
	}
}
