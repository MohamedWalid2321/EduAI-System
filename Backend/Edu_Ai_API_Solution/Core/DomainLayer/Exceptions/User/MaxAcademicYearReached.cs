using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.User
{
	public sealed class MaxAcademicYearReached() :BadRequestException("Max Academic Year Reached. Cannot enroll in more than level 5.")
	{
	}
}
