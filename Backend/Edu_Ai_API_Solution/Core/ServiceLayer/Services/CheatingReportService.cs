using DomainLayer.Models;
using Mapster;
using ServiceLayer.Specifications.AttemptedQuizSpecification;
using ServiceLayer.Specifications.CheatingReportSpecifications;
using Shared.Dtos.CheatingReportDto.Request;
using Shared.Dtos.CheatingReportDto.Response;

namespace ServiceLayer.Services
{
    public class CheatingReportService(IUnitOfWork _unitOfWork) : ICheatingReportService
    {
        public async Task<CheatingReportResponse> GetByAttemptIdAsync(int attemptId, CancellationToken cancellationToken = default)
        {
            var repo = _unitOfWork.GetRepository<CheatingReport, int>();
            var spec = new CheatingReportByAttemptSpecification(attemptId);
            var report = await repo.GetFirstOrDefaultAsync(spec, cancellationToken)
                ?? throw new KeyNotFoundException($"No cheating report found for attempt {attemptId}");

            return MapToResponse(report);
        }

        public async Task<IEnumerable<CheatingReportResponse>> GetByQuizIdAsync(int quizId, CancellationToken cancellationToken = default)
        {
            var repo = _unitOfWork.GetRepository<CheatingReport, int>();
            var spec = new CheatingReportsByQuizSpecification(quizId);
            var reports = await repo.GetAllAsync(spec, cancellationToken);

            return reports.Select(MapToResponse);
        }

        public async Task<CheatingReportResponse> CreateAsync(int attemptId, CancellationToken cancellationToken = default)
        {
            var attemptRepo = _unitOfWork.GetRepository<QuizAttempt, int>();
            var attempt = await attemptRepo.GetFirstOrDefaultAsync(
                new AttemptWithUserSpecification(attemptId), cancellationToken)
                ?? throw new KeyNotFoundException($"Quiz attempt {attemptId} not found");

            var repo = _unitOfWork.GetRepository<CheatingReport, int>();

            // Enforce one report per attempt
            var existing = await repo.GetFirstOrDefaultAsync(
                new CheatingReportByAttemptSpecification(attemptId), cancellationToken);

            if (existing is not null)
                return MapToResponse(existing);

            var report = new CheatingReport
            {
                QuizAttemptId = attemptId,
                StudentId = attempt.StudentId,
                Student = attempt.User   // carry the loaded user so MapToResponse has it
            };

            await repo.AddAsync(report, cancellationToken);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            return MapToResponse(report);
        }

        public async Task<CheatingViolationResponse> AddViolationAsync(int reportId, AddViolationRequest request, CancellationToken cancellationToken = default)
        {
            var reportRepo = _unitOfWork.GetRepository<CheatingReport, int>();
            var report = await reportRepo.GetByIdAsync(reportId, cancellationToken)
                ?? throw new KeyNotFoundException($"Cheating report {reportId} not found");

            var violation = new CheatingViolation
            {
                CheatingReportId = reportId,
                EvidenceUrl = request.EvidenceUrl,
                Timestamp = request.Timestamp,
                Description = request.Description
            };

            var violationRepo = _unitOfWork.GetRepository<CheatingViolation, int>();
            await violationRepo.AddAsync(violation, cancellationToken);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            return violation.Adapt<CheatingViolationResponse>();
        }

        public async Task DeleteViolationAsync(int violationId, CancellationToken cancellationToken = default)
        {
            var repo = _unitOfWork.GetRepository<CheatingViolation, int>();
            var violation = await repo.GetByIdAsync(violationId, cancellationToken)
                ?? throw new KeyNotFoundException($"Violation {violationId} not found");

            repo.Delete(violation);
            await _unitOfWork.SaveChangesAsync(cancellationToken);
        }

        public async Task DeleteReportAsync(int reportId, CancellationToken cancellationToken = default)
        {
            var repo = _unitOfWork.GetRepository<CheatingReport, int>();
            var report = await repo.GetByIdAsync(reportId, cancellationToken)
                ?? throw new KeyNotFoundException($"Cheating report {reportId} not found");

            repo.Delete(report);
            await _unitOfWork.SaveChangesAsync(cancellationToken);
        }

        public async Task<RiskAssessmentResultResponse> GetRiskAssessmentByCheatingReportAsync(int cheatingReportId, CancellationToken cancellationToken = default)
        {
            var repo = _unitOfWork.GetRepository<RiskAssessmentResult, int>();
            var spec = new RiskAssessmentResultByCheatingReportSpecification(cheatingReportId);
            var result = await repo.GetFirstOrDefaultAsync(spec, cancellationToken)
                ?? throw new KeyNotFoundException($"No risk assessment result found for cheating report {cheatingReportId}");

            return new RiskAssessmentResultResponse
            {
                Id                      = result.Id,
                StudentId               = result.StudentId,
                AttemptId               = result.AttemptId,
                CheatingReportId        = result.CheatingReportId,
                SessionViolationRate    = result.SessionViolationRate,
                OverallSessionRiskScore = result.OverallSessionRiskScore,
                Questions = result.Questions.Select(q => new RiskQuestionResultDto
                {
                    QuestionId         = q.QuestionId,
                    StudentRiskScore   = q.StudentRiskScore,
                    CohortAvgRiskScore = q.CohortAvgRiskScore
                }).ToList()
            };
        }

        private static CheatingReportResponse MapToResponse(CheatingReport report) => new()
        {
            Id = report.Id,
            QuizAttemptId = report.QuizAttemptId,
            StudentId = report.StudentId,
            StudentName = report.Student is not null
                ? $"{report.Student.FirstName} {report.Student.LastName}"
                : string.Empty,
            Violations = report.Violations?
                .Select(v => v.Adapt<CheatingViolationResponse>())
                .ToList() ?? []
        };
    }
}