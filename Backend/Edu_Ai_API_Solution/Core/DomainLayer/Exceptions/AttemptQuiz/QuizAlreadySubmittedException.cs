using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AttemptQuiz
{
    public  class QuizAlreadySubmittedException(string studentId) : ConflictException($"This Quiz has already been submitted for student with ID : {studentId}.")
    {
    }
}
