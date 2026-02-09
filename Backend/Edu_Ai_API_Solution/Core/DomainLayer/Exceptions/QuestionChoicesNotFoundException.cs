using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions
{
    public sealed class QuestionChoicesNotFoundException(int id) : NotFoundException($"Question Choices with this Id : {id} is Not Found ")
    {
    }
}
