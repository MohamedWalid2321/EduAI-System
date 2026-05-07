namespace Shared.Dtos.CheatingReportDto.Response
{
    public class CheatingReportResponse
    {
        public int Id { get; set; }
        public int QuizAttemptId { get; set; }
        public string StudentId { get; set; }
        public string StudentName { get; set; }
        public List<CheatingViolationResponse> Violations { get; set; } = [];
    }
}