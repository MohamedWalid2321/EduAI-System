using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.User
{
	public class IsNotInstructorException(string instructorId) :ConflictException($"User {instructorId} is not an instructor.")
	{
	}
}
