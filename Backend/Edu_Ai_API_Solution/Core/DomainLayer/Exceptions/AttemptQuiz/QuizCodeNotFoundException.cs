using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AttemptQuiz
{
    public sealed class QuizCodeNotFoundException(string code): NotFoundException($"Quiz with code '{code}' was not found.")
    {
    }
}
