using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.InstructorCourseSpecifications
{
	public class InstructorCourseSpecification : BaseSpecification<InstructorCourse, int>
	{
		// Get all instructors for a course
		public InstructorCourseSpecification(int courseId) 
			: base(ic => ic.CourseId == courseId)
		{
			AddInclude(ic => ic.Instructor);
			AddInclude(ic => ic.Course);
		}
		
		// Check if specific instructor is enrolled in a course
		public InstructorCourseSpecification(int courseId, string instructorId) 
			: base(ic => ic.CourseId == courseId && ic.InstructorId == instructorId)
		{
			AddInclude(ic => ic.Instructor);
			AddInclude(ic => ic.Course);
		}
	}
}