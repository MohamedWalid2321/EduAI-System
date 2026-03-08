using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AttemptQuiz
{
    public sealed class QuizAlreadyAttemptedException(string quizCode, string studentId) : ConflictException($"Student with id {studentId} has already attempted quiz with id {quizCode}.")
    { 
    }
    
}
