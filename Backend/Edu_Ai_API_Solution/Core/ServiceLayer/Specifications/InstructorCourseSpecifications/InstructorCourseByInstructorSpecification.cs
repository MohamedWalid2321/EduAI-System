using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.InstructorCourseSpecifications
{
	public class InstructorCourseByInstructorSpecification : BaseSpecification<InstructorCourse, int>
	{
		public InstructorCourseByInstructorSpecification(string instructorId) 
			: base(ic => ic.InstructorId == instructorId)
		{
			AddInclude(ic => ic.Course);
			AddInclude(ic => ic.Instructor);
		}
	}
}