using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Quiz
{
    public sealed class QuizzesInCourseNotFoundException(int id) : NotFoundException($"Quizzes in Course with this Id : {id} is Not Found ")
    {
    }
}
