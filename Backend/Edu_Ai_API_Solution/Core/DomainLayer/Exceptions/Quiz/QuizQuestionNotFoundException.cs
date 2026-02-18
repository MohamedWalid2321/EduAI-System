using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Quiz
{
    public sealed class QuizQuestionNotFoundException(int id) : NotFoundException($"Quiz Question with this Id : {id} is Not Found ")
    {
    }
}
