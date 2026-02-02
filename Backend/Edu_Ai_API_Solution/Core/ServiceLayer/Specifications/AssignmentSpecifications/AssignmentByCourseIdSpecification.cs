using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AssignmentSpecifications
{
	public class AssignmentByCourseIdSpecification : BaseSpecification<Assignment, int>
	{
		public AssignmentByCourseIdSpecification(int courseId) : base(a => a.CourseId == courseId)
		{
			AddInclude(p => p.AssignmentAttachments);
		}
	}
}