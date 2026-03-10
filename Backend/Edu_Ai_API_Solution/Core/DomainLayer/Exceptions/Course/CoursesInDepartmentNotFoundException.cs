using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Course
{
    public sealed class CoursesInDepartmentNotFoundException(int id): NotFoundException($"There is no Courses in Department with id :{id}")
    {
    }
}
