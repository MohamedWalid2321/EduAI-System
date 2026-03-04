using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Shared.Dtos.DepartmentDto.Request
{
	public class DepartmentRequest {
		public string Title { get; set; } = null!;
	}
}
