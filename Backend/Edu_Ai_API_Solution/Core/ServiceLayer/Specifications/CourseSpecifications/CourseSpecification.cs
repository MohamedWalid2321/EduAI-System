using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.CourseSpecification
{
	public class CourseSpecification: BaseSpecification<Course,int>
	{
		public CourseSpecification(int departmentId, int id) : base(c => c.Id == id && c.Departments.Any(d=>d.Id==departmentId))
		{
			AddInclude(c => c.Assessments);
			AddInclude(c => c.Departments);
		}
		public CourseSpecification( int id) : base(c => c.Id == id)
		{
			AddInclude(c => c.Assessments);
			AddInclude(c => c.Departments);
		}
		public CourseSpecification() : base(null)
		{
			AddInclude(c => c.Assessments);
			AddInclude(c => c.Departments);
		}
	}
}
