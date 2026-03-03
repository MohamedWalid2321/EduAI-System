using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.DepartmentSpecification
{
	public class DepartmentByTitleSpecification : BaseSpecification<Department, int>
	{
		public DepartmentByTitleSpecification(string title) : base(p => p.Title.ToLower() == title.ToLower())
		{
		}
	}
}
