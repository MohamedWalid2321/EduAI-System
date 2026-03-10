using DomainLayer.Enums;
using Microsoft.AspNetCore.Identity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public sealed class ApplicationUser:IdentityUser
	{
		public string? FirstName { get; set; } = string.Empty;
		public string? LastName { get; set; }= string.Empty;
		public string? ProfilePictureUrl { get; set; } = string.Empty;
		public string? ProfilePictureBase64 { get; set; } = string.Empty;
		public DateOnly DateOfBirth { get; set; }
		public bool IsDisabled { get; set; }
		public List<RefreshToken> RefreshTokens { get; set; } = [];

		public int? DepartmentId { get; set; }
		public Department? Department { get; set; }
		public AcademicYear? AcademicYear { get; set; }
		public bool IsEnrolled { get; set; } = false;
		public DateTime? EnrolledAt { get; set; }

        // Add this property to the existing ApplicationUser class
        public ICollection<InstructorCourse> InstructorCourses { get; set; } = new List<InstructorCourse>();
	}
}
