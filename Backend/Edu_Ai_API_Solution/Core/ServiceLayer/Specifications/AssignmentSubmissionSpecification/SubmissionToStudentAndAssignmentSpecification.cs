using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AssignmentSubmissionSpecification
{
    public class SubmissionToStudentAndAssignmentSpecification : BaseSpecification<AssignmentSubmission,int>
    {
        public SubmissionToStudentAndAssignmentSpecification(string studentId, int assignmentId)
            : base(s => s.StudentId == studentId && s.AssignmentId == assignmentId)
        {
             
        }
    }
}
