using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Department
{
	public sealed class DepartmentIdNotFoundInToken():UnAuthorizedException("Department ID not found in token , You are not Student Or Instructor")
	{
	}
}
