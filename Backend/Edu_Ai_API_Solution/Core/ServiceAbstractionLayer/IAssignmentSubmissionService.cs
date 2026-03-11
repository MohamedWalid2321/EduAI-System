using Shared.Dtos.AssignmentSubmissionDto.Request;
using Shared.Dtos.AssignmentSubmissionDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IAssignmentSubmissionService
    {
        Task<AssignmentSubmissionResponseDto> SubmitAssignmentAsync(string studentId,AssignmentSubmissionRequestDto request, List<IFormFile?> Files);
        Task<AssignmentSubmissionResponseDto> GetSubmissionByIdAsync(int submissionId);
        Task<IEnumerable<AssignmentSubmissionResponseDto>> GetSubmissionsByAssignmentIdAsync(int assignmentId);
        Task<IEnumerable<AssignmentSubmissionResponseDto>> GetSubmissionsByStudentIdAsync(string studentId);
        Task DeleteSubmissionAsync(int submissionId);
        Task<AssignmentSubmissionResponseDto> GradeSubmissionAsync(int submissionId, GradeAssignmentSubmissionRequestDto request);
    }
}
