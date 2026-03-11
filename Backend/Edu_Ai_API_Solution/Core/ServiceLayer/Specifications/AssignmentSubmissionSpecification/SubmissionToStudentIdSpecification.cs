using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AssignmentSubmissionSpecification
{
    public class SubmissionToStudentIdSpecification : BaseSpecification<AssignmentSubmission, int>
    {
        public SubmissionToStudentIdSpecification(string studentId) : base(a => a.StudentId == studentId)
        {
                AddInclude(b => b.AssignmentSubmissionAttachments);
        }
    }
}
