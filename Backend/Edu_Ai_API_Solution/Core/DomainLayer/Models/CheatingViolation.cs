namespace DomainLayer.Models
{
    public class CheatingViolation : BaseEntity<int>
    {
        public string EvidenceUrl { get; set; }        // Bunny CDN URL (mp4 or mp3)
        public DateTime Timestamp { get; set; }        // When the violation occurred
        public string Description { get; set; }        // What happened in this clip

        public int CheatingReportId { get; set; }
        public CheatingReport CheatingReport { get; set; }
    }
}