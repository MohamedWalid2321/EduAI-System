using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Shared.Dtos.CourseDto.Request
{
	public class CourseRequestDto {
		public int Id { get; set; }
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public string semster { get; set; } = string.Empty;
		public int Credit_Hour { get; set; }
		public string CourseStatus { get; set; } = null!;
		public string LearningOutcomes { get; set; } = null!;
		public ICollection<AssesmentDto> Assesment { get; set; } = null!;

	}
}
