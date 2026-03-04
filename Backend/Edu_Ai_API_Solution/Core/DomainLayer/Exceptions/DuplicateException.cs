using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions
{
    public abstract class DuplicateException(string message): Exception(message)
    {
    }
}
