using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Question
{
    public sealed class InvalidCorrectAnswerIndexException(): BadRequestException("The correct answer index is invalid. It must be a non-negative integer and less than the number of choices.")
    {
    }
}
