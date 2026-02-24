using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Assesment
{
    public sealed class AssessmentNotFoundException(int id): NotFoundException($"Assessment with this Id : {id} is Not Found ")
    {
    }
}
