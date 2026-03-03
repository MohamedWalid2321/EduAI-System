using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Department
{
	public sealed class DuplicatedDepartmentException():ConflictException("Department with the same title already exists.")
	{
	}
}
