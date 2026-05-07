namespace Shared.Dtos.CheatingReportDto.Request
{
    public class AddViolationRequest
    {
        public string EvidenceUrl { get; set; }   // Bunny CDN mp4 or mp3 URL
        public DateTime Timestamp { get; set; }
        public string Description { get; set; }
    }
}