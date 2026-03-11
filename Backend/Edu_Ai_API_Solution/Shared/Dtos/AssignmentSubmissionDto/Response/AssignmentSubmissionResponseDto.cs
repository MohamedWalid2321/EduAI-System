using Shared.Dtos.AssigmentDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AssignmentSubmissionDto.Response
{
    public class AssignmentSubmissionResponseDto
    {
        public int Id { get; set; }
        public int AssignmentId { get; set; }
        public string StudentId { get; set; }
        public string? TextSubmission { get; set; }
        public DateTime SubmittedAt { get; set; }
        public int? Grade { get; set; }
        public string? Feedback { get; set; }
        public List<AssignmentSubmissionAttachmentDto> AssignmentSubmissionAttachments { get; set; } = [];

    }
}
