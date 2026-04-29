using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.CourseSpecifications
{
	public class StudentCourseSpecification: BaseSpecification<Course, int>
	{
		public StudentCourseSpecification(int? departmentId, AcademicYearEnum? academicYearEnum) : base(c => c.Departments.Any(d => d.Id == departmentId) && c.AcademicLevel == academicYearEnum && c.IsPublished==true)
		{
			AddInclude(c => c.Assessments);
			AddInclude(c => c.Departments);
		}
	}
}
