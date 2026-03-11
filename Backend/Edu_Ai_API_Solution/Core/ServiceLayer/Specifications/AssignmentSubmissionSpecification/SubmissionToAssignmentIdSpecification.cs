using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AssignmentSubmissionSpecification
{
    public class SubmissionToAssignmentIdSpecification :BaseSpecification<AssignmentSubmission, int>
    {
        public SubmissionToAssignmentIdSpecification(int assignmentId) :base(a=>a.AssignmentId ==  assignmentId) {
            AddInclude(b => b.AssignmentSubmissionAttachments);
        }
    }
}
