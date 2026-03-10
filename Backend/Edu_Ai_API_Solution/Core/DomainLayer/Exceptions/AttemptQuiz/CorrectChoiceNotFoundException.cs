using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AttemptQuiz
{
    public class SelectedChoiceNotFoundException(int questionId):
        NotFoundException($"Selected choice for question with id {questionId} was not found.")
    {
    }
}
