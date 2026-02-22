using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AuthDto.Request
{
	public class ConfirmEmailRequest
	{
		public string userId { get; set; } = null!;
		public string code { get; set; } = null!;
	}
}
