using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Assigment
{
    public sealed class AssignmentNotFoundException(int id) : NotFoundException($"Assignment with This Id : {id} is Not Found ")
    {
        
    }
}
