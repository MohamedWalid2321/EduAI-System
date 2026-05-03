using DomainLayer.Enums;
using Microsoft.AspNetCore.Identity;

namespace DomainLayer.Models
{
	public sealed class ApplicationUser : IdentityUser
	{
		public string? FirstName { get; set; } = string.Empty;
		public string? LastName { get; set; } = string.Empty;
		public string? ProfilePictureUrl { get; set; } = string.Empty;
		public DateOnly DateOfBirth { get; set; }
		public bool IsDisabled { get; set; }
		public List<RefreshToken> RefreshTokens { get; set; } = [];
		public int AcademicYearId { get; set; }
		public int? DepartmentId { get; set; }
		public Department? Department { get; set; }
		public AcademicYearEnum? AcademicYearEnum { get; set; }
		public bool IsEnrolled { get; set; } = false;
		public DateTime? EnrolledAt { get; set; }

		public ICollection<UserCourse> UserCourses { get; set; } = [];
		public NotificationBox? NotificationBox { get; set; }
	}
}
