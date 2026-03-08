using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AttemptQuiz
{
    public class QuizNotSubmittedException(string studentId): BadRequestException($"Student with id {studentId} has not submitted the quiz yet.")
    {
    }
}
