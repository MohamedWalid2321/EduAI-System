using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AttemptQuiz
{
    public class QuizTimeExpiredException():BadRequestException("you exceeded the allowed quiz duration.")
    {
    }
}
