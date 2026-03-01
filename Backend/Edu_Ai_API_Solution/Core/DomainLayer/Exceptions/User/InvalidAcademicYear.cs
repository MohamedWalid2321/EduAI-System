using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.User
{
	public sealed class InvalidAcademicYear():BadRequestException("Invalid Academic Year. Must be: First, Second, Third, Fourth, or Fifth")
	{
	}
}
