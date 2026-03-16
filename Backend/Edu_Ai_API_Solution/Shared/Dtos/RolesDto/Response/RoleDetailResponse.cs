using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.RolesDto.Response
{
	public class RoleDetailResponse
	{
		public string Id { get; set; } = string.Empty;
		public string Name { get; set; } = string.Empty;
		public bool IsDeleted { get; set; }
		public bool IsEnrollable { get; set; }
		public IList<string> Permissions { get; set; } = [];
	}
}
