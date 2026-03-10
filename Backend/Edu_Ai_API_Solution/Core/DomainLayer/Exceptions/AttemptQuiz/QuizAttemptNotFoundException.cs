using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AttemptQuiz
{
    public  class QuizAttemptNotFoundException(string studentId) :NotFoundException($"Quiz Attempt Not Found to student with id {studentId}")
    {
    }
}
