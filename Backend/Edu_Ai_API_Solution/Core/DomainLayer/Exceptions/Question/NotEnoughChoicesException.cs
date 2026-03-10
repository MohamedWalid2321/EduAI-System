using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Question
{
    public sealed class NotEnoughChoicesException(): BadRequestException($"A question must have at least two choices")
    {
    }
}
