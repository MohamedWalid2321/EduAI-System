using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Question
{
    public sealed class QuestionAlreadyExistsException(string title) : ConflictException($"Question with this Content : {title} already exists")
    {
    }
}
