using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Claims;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Extensions
{
	public static class UserExtensions
	{
		public static string? GetUserId(this ClaimsPrincipal user) =>
			user.FindFirstValue(ClaimTypes.NameIdentifier);
	}
}
