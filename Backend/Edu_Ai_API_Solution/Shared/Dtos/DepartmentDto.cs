using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Shared.Dtos
{
	public class DepartmentDto {
		public int Id { get; set; }
		public string Title { get; set; } = null!;
		
		[JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
		public ICollection<CourseDtoClass>? courses { get; set; }
	}
}
