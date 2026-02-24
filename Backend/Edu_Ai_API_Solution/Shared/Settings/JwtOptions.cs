using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Settings
{
	public class JwtOptions
	{
		public static string SectionName = "jwt";
		[Required]
		public string Key { get; set; } = string.Empty;
		[Required]
		public string Issuer { get; set; } = string.Empty;
		[Required]
		public string Audience { get; set; } = string.Empty;
		[Range(60, int.MaxValue, ErrorMessage = "invalid expired seconds")]
		public int ExpiredSeconds { get; set; }
	}
}
