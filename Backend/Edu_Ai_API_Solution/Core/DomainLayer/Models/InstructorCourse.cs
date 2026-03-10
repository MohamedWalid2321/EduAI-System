using System;

namespace DomainLayer.Models
{
	public class InstructorCourse : BaseEntity<int>
	{
		public string InstructorId { get; set; } = string.Empty;
		public ApplicationUser Instructor { get; set; } = null!;
		
		public int CourseId { get; set; }
		public Course Course { get; set; } = null!;
		
		public DateTime AssignedAt { get; set; } = DateTime.UtcNow;
		public string? AssignedBy { get; set; } // Admin who assigned the instructor
	}
}