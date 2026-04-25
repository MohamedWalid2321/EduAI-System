using DomainLayer.Enums;

namespace DomainLayer.Models
{
	public class UserCourse : BaseEntity<int>
	{
		public string UserId { get; set; } = null!;
		public ApplicationUser User { get; set; } = null!;

		public int CourseId { get; set; }
		public Course Course { get; set; } = null!;

		public DateTime EnrolledAt { get; set; } = DateTime.UtcNow;
		public EnrollmentStatus Status { get; set; } = EnrollmentStatus.Active;
		public string? EnrolledBy { get; set; }
	}
}
