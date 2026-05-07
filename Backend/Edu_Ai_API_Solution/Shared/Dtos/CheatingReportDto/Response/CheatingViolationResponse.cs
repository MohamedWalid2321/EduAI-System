namespace Shared.Dtos.CheatingReportDto.Response
{
    public class CheatingViolationResponse
    {
        public int Id { get; set; }
        public string EvidenceUrl { get; set; }
        public DateTime Timestamp { get; set; }
        public string Description { get; set; }
    }
}