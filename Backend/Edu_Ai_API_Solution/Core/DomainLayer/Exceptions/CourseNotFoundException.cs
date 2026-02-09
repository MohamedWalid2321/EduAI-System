using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions
{
    public sealed class CourseNotFoundException (int id) : NotFoundException($"Course with this Id : {id} is Not Found ")
    {
    }
}
