using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.CourseDto.Response
{
	public class CourseResponseDto
	{
		public int Id { get; set; }
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public string ImageUrl { get; set; } = null!;
		public string semster { get; set; } = string.Empty;
		public int Credit_Hour { get; set; }
		public bool IsPublished { get; set; }
		public string LearningOutcomes { get; set; } = null!;
	}
}
