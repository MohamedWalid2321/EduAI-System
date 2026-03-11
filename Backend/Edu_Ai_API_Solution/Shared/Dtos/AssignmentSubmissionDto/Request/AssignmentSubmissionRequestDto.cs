using Shared.Dtos.AssigmentDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AssignmentSubmissionDto.Request
{
    public class AssignmentSubmissionRequestDto
    {
        public int AssignmentId { get; set; }
        public string TextSubmission { get; set; } = string.Empty;
        //public List<AssignmentSubmissionAttachmentDto> AssignmentSubmissionAttachments { get; set; } = [];
    }
}
