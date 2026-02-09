using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions
{
    public sealed class AssignmentInCourseNotFoundException(int id): NotFoundException($"Assignments in Course with this Id : {id} is Not Found ")
    {
    }
}
