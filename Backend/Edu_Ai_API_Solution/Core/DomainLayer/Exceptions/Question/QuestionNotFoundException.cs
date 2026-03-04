using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Question
{
    public sealed class QuestionNotFoundException(int id): NotFoundException($"Question with this Id : {id} is Not Found ")
    {
    }
}
