using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Question
{
    public sealed class TooManyChoicesException(): BadRequestException($"Question should have at least 2 choices and at most 4 choices")
    {
    }
}
