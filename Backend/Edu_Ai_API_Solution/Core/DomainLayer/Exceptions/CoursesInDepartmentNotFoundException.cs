using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions
{
    public sealed class CoursesInDepartmentNotFoundException(int id): NotFoundException($"Courses in Department with this Id : {id} is Not Found ")
    {
    }
}
