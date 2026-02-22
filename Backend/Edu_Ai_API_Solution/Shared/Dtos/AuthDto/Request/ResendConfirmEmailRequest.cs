using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AuthDto.Request
{
	public class ResendConfirmEmailRequest
	{
		public string Email { get; set; } = null!;
	}
}
