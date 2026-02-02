using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AssignmentSpecifications
{
	public class AssignmentSpecification : BaseSpecification<Assignment, int>
	{
		public AssignmentSpecification(int id) : base(p => p.Id == id)
		{
			AddInclude(p => p.AssignmentAttachments);
		}
		
		public AssignmentSpecification() : base(null)
		{
			AddInclude(p => p.AssignmentAttachments);
		}
	}
}