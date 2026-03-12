using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AssignmentSubmission
{
    public sealed class AssignmentSubmissionNotFoundException(int submissionId) : NotFoundException($"Submission with ID {submissionId} not found.")
    {
    }
}
