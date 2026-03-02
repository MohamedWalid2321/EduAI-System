using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Settings
{
	public class MailSettings
	{
		public string Mail { get; set; } = string.Empty;
		public string DisplayName { get; set; } = string.Empty;
		public string Password { get; set; } = string.Empty;
		public string Host { get; set; } = string.Empty;
		public int Port { get; set; }
	}
}
