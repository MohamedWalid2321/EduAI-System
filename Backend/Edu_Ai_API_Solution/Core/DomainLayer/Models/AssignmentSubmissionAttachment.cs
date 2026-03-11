using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
    public class AssignmentSubmissionAttachment:BaseEntity<Guid>
    {
        public AssignmentSubmissionAttachment() { 
            Id = Guid.NewGuid();
        }

        public string FileName { get; set; } = null!; // title of the file
        public string FileUrl { get; set; } = null!;
        public string Type { get; set; } = null!; // e.g., "application/pdf", "image/png"
        // AssignmentSubmission RelationShip
        public int AssignmentSubmissionId { get; set; }
        public AssignmentSubmission AssignmentSubmission { get; set; }

    }
}
