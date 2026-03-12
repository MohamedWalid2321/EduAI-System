using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AssignmentSubmissionDto.Request
{
    public class GradeAssignmentSubmissionRequestDto
    {
        public int Grade { get; set; }
        public string Feedback { get; set; }
    }
}
