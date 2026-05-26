using DomainLayer.Exceptions.AssignmentSubmission;
using ServiceLayer.Specifications.AssignmentSubmissionSpecification;
using Shared.Dtos.AssignmentSubmissionDto.Request;
using Shared.Dtos.AssignmentSubmissionDto.Response;

namespace ServiceLayer.Services
{
    public class AssignmentSubmissionService(IUnitOfWork _unitOfWork , IFileStorageService _fileStorageService) : IAssignmentSubmissionService
    {
        public async Task<AssignmentSubmissionResponseDto> GetSubmissionByIdAsync(int submissionId, CancellationToken cancellationToken = default)
        {
            var submissionRepo = _unitOfWork.GetRepository<AssignmentSubmission, int>();

            var submissionSpecification = new SubmissionWithAttachmentsSpecification(submissionId);


            var submission = await submissionRepo.GetByIdAsync(submissionSpecification, cancellationToken);

            if (submission == null)
            {
                throw new AssignmentSubmissionNotFoundException(submissionId);
            }
            return submission.Adapt<AssignmentSubmissionResponseDto>();
        }

        public async Task<IEnumerable<AssignmentSubmissionResponseDto>> GetSubmissionsByAssignmentIdAsync(int assignmentId, CancellationToken cancellationToken = default)
        {
            var submissionRepo = _unitOfWork.GetRepository<AssignmentSubmission, int>();
            var assignmentRepo = _unitOfWork.GetRepository<Assignment, int>();

            var assignment = await assignmentRepo.GetByIdAsync(assignmentId, cancellationToken);

            if (assignment is null)
                throw new AssignmentNotFoundException(assignmentId);

            var submissionSpecification = new SubmissionToAssignmentIdSpecification(assignmentId);
            var submissions = await submissionRepo.GetAllAsync(submissionSpecification, cancellationToken);

            return submissions.Adapt<IEnumerable<AssignmentSubmissionResponseDto>>();
        }

        public async Task<IEnumerable<AssignmentSubmissionResponseDto>> GetSubmissionsByStudentIdAsync(string studentId, CancellationToken cancellationToken = default)
        {
            var submissionRepo = _unitOfWork.GetRepository<AssignmentSubmission, int>();
            var submissionSpecification = new SubmissionToStudentIdSpecification(studentId);

            var submissions = await submissionRepo.GetAllAsync(submissionSpecification, cancellationToken);

            var response = submissions.Adapt<IEnumerable<AssignmentSubmissionResponseDto>>();
            return response;
        }

        public async Task<AssignmentSubmissionResponseDto> SubmitAssignmentAsync(string studentId, AssignmentSubmissionRequestDto request, List<IFormFile?> Files, CancellationToken cancellationToken = default)
        {
            var assignmentRepo = _unitOfWork.GetRepository<Assignment, int>();  
            var submissionRepo = _unitOfWork.GetRepository<AssignmentSubmission, int>();
            var submissionAttachmentRepo = _unitOfWork.GetRepository<AssignmentSubmissionAttachment, Guid>();

            var assignmentId = request.AssignmentId;
            var assignment = await assignmentRepo.GetByIdAsync(assignmentId, cancellationToken);

            if (assignment is null)
                throw new AssignmentNotFoundException(assignmentId);

            var exitingSubmissionSpecification = new SubmissionToStudentAndAssignmentSpecification(studentId, assignmentId);

            var existingSubmission = await submissionRepo.GetFirstOrDefaultAsync(exitingSubmissionSpecification, cancellationToken);

            if (existingSubmission != null)
            {
                throw new DuplicateSubmissionException(studentId, assignmentId);
            }

            if(assignment.DueDate < DateTime.UtcNow)
            {
                throw new AssignmentDueDatePassedException(assignmentId);
            }   

            var submission = new AssignmentSubmission
            {
                AssignmentId = request.AssignmentId,
                StudentId = studentId, 
                TextSubmission = request.TextSubmission,
                SubmittedAt = DateTime.UtcNow,
            };

            await submissionRepo.AddAsync(submission, cancellationToken);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            foreach (var file in Files)
            {
                if (file is not null && file.Length > 0)
                {
                    using var stream = file.OpenReadStream();
                    var fileUrl = await _fileStorageService.UploadFileAsync(
                        stream,
                        file.FileName,
                        $"assignmentSubmissions/{submission.Id}/attachments",
                        file.ContentType
                    );

                    var attachment = new AssignmentSubmissionAttachment
                    {
                        FileName = file.FileName,
                        FileUrl = fileUrl,
                        Type = file.ContentType,
                        AssignmentSubmissionId = submission.Id
                    };

                    await submissionAttachmentRepo.AddAsync(attachment, cancellationToken);
                }
            }
            await _unitOfWork.SaveChangesAsync(cancellationToken);
            return submission.Adapt<AssignmentSubmissionResponseDto>();
        }

        public async Task DeleteSubmissionAsync(int submissionId, CancellationToken cancellationToken = default)
        {
            var submissionRepo = _unitOfWork.GetRepository<AssignmentSubmission, int>();

            var submission = await submissionRepo.GetByIdAsync(submissionId, cancellationToken);

            if (submission == null)
                throw new AssignmentSubmissionNotFoundException(submissionId);

            submissionRepo.HardDelete(submission);
            await _unitOfWork.SaveChangesAsync(cancellationToken);
        }

        public async Task<AssignmentSubmissionResponseDto> GradeSubmissionAsync(int submissionId, GradeAssignmentSubmissionRequestDto request, CancellationToken cancellationToken = default)
        {
            var submissionRepo = _unitOfWork.GetRepository<AssignmentSubmission, int>();
            var assignmentRepo = _unitOfWork.GetRepository<Assignment, int>();

            

            var submission = await submissionRepo.GetByIdAsync(submissionId, cancellationToken);
            if (submission == null)
                throw new AssignmentSubmissionNotFoundException(submissionId);

            var assignment = await assignmentRepo.GetByIdAsync(submission.AssignmentId, cancellationToken);

            if (request.Grade < 0 || request.Grade > assignment?.TotalMarks)
                throw new InvalidGradeException(request.Grade);

            submission.Grade = request.Grade;
            submission.Feedback = request.Feedback;


            submissionRepo.Update(submission);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            var submissionSpecification = new SubmissionWithAttachmentsSpecification(submissionId);


            var entity = await submissionRepo.GetByIdAsync(submissionSpecification, cancellationToken);

            return entity.Adapt<AssignmentSubmissionResponseDto>();
        }
    }
}
