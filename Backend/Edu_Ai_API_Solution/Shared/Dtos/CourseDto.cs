using Shared.Dtos.AssesmentDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos
{
	public class CourseDtoClass
	{
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public string ImageUrl { get; set; } = null!;
		public string semster { get; set; } = string.Empty;
		public int Credit_Hour { get; set; }
		public string CourseStatus { get; set; } = null!;
		public string LearningOutcomes { get; set; } = null!;
		public AssesmentRequest Assesment { get; set; } = null!;
	}
}
