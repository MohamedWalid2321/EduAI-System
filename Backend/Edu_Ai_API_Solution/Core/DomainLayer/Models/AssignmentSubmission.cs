using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
    public class AssignmentSubmission :BaseEntity<int>
    {
        public int AssignmentId { get; set; }
        public Assignment? Assignment { get; set; }
        public string StudentId { get; set; }
        public string? TextSubmission { get; set; }
        public DateTime SubmittedAt { get; set; }
        public int? Grade { get; set; }
        public string? Feedback { get; set; }
        public ICollection<AssignmentSubmissionAttachment> AssignmentSubmissionAttachments { get; set; }
    }
}
