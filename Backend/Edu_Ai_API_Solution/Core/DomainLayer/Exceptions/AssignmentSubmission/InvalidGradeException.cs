using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AssignmentSubmission
{
    public sealed class InvalidGradeException(int grade):BadRequestException($"The grade '{grade}' is invalid.")
    {
    }
}
