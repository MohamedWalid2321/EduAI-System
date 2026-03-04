using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.CourseDto.Response
{
	public class FullCourseResponse
	{
		public int Id { get; set; }
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public string ImageUrl { get; set; } = null!;
		public string semster { get; set; } = string.Empty;
		public int Credit_Hour { get; set; }
		public string CourseStatus { get; set; } = null!;
		public string LearningOutcomes { get; set; } = null!;
		public List<AssesmentDto> Assesment { get; set; } = [];
	}
}
