namespace Shared.Dtos.AssesmentDto.AssesmentDto
{
    public class AssessmentResponseDto
    {
        public int Id { get; set; }
        public string AssType { get; set; }
        public double PercentageWeight { get; set; }
        public bool IsMandatory { get; set; }
        public int Hours { get; set; }
        public int CourseId { get; set; }
    }
}