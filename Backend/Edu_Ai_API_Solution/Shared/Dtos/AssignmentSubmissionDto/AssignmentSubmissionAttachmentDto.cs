using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AssignmentSubmissionDto
{
    public class AssignmentSubmissionAttachmentDto
    {
        public Guid Id { get; set; } 
        public string FileName { get; set; } = null!;
        public string FileUrl { get; set; } = null!;
        public string Type { get; set; } = null!;
    }
}
