using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AuthDto.Request
{
	public class ForgetPasswordRequest
	{
		public string Email { get; set; } = null!;
	}
}
