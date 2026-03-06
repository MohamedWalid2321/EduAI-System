using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.CourseSpecification
{
	public class CourseByDepartmentSpecification: BaseSpecification<Course,int>
	{
		public CourseByDepartmentSpecification(int departmentId) : base(c => c.Departments.Any(d=>d.Id==departmentId) && c.IsPublished==true)
		{
			AddInclude(c => c.Assessments);
		}
	}
}
