using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AssignmentSubmission
{
    public sealed class AssignmentDueDatePassedException(int id):BadRequestException($"Assignment {id} due date has already passed.")
    {
    }
}
