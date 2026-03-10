using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.User
{
	public class DuplicatedInstructorEnrollmentException(string instructorId,int courseId) :ConflictException($"Instructor {instructorId} is already enrolled in course {courseId}.")
	{
	}
}
