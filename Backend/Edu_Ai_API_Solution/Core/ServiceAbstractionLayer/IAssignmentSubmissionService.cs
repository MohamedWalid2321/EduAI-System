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
        Task<AssignmentSubmissionResponseDto> SubmitAssignmentAsync(string studentId, AssignmentSubmissionRequestDto request, List<IFormFile?> Files, CancellationToken cancellationToken = default);
        Task<AssignmentSubmissionResponseDto> GetSubmissionByIdAsync(int submissionId, CancellationToken cancellationToken = default);
        Task<IEnumerable<AssignmentSubmissionResponseDto>> GetSubmissionsByAssignmentIdAsync(int assignmentId, CancellationToken cancellationToken = default);
        Task<IEnumerable<AssignmentSubmissionResponseDto>> GetSubmissionsByStudentIdAsync(string studentId, CancellationToken cancellationToken = default);
        Task DeleteSubmissionAsync(int submissionId, CancellationToken cancellationToken = default);
        Task<AssignmentSubmissionResponseDto> GradeSubmissionAsync(int submissionId, GradeAssignmentSubmissionRequestDto request, CancellationToken cancellationToken = default);
    }
}
