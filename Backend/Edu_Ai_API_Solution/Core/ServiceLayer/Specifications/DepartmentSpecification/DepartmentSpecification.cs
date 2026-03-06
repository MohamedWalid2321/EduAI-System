using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.DepartmentSpecification
{
	public class DepartmentSpecification : BaseSpecification<Department, int>
	{
		public DepartmentSpecification(int id) : base(p=> p.Id==id)
		{
			AddInclude(p => p.courses);
		}
		public DepartmentSpecification(): base(null)
		{
			AddInclude(p => p.courses);
		}

	}
}
