using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Question
{
    public sealed class DuplicateChoicesException(): ConflictException("Duplicate choices are not allowed in a question.")
    {
    }
}
