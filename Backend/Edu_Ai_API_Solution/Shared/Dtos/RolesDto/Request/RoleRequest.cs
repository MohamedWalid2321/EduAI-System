using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.RolesDto.Request
{
	public class RoleRequest
	{
		public string Name { get; set; }= string.Empty;
		public bool IsEnrollable { get; set; }
		public IList<string> Permissions { get; set; } = [];
	}
}
