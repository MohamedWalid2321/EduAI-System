using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AssignmentSubmission
{
    public sealed class DuplicateSubmissionException(string studentId, int assignmentId) : ConflictException($"Student with ID '{studentId}' has already submitted for Assignment with ID '{assignmentId}'.")
    {
    }
}
