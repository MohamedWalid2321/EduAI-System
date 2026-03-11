using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AssignmentSubmissionSpecification
{
    public class SubmissionWithAttachmentsSpecification : BaseSpecification<AssignmentSubmission, int>
    {
        public SubmissionWithAttachmentsSpecification(int submissionId) : base(s => s.Id == submissionId)
        {
                AddInclude(s => s.AssignmentSubmissionAttachments);
        }
    }
}
