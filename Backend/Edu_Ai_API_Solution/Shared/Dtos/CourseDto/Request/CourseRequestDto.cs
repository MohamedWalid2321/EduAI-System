using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Shared.Dtos.CourseDto.Request
{
	public class CourseRequestDto {
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		[Range(1,3)]
		public int semster { get; set; }
		public int Credit_Hour { get; set; }
		public string LearningOutcomes { get; set; } = null!;
	}
}
