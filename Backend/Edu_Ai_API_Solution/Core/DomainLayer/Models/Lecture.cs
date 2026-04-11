using System;

namespace DomainLayer.Models
{
	public class Lecture : BaseEntity<int>
	{
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public string RoomName { get; set; } = null!;           // Jitsi unique room name
		public DateTime ScheduledAt { get; set; }
		public bool IsActive { get; set; } = false;             // Instructor toggles this to open/close the room

		// Course Relationship
		public int CourseId { get; set; }
		public Course Course { get; set; } = null!;

		// Created by
		public string CreatedById { get; set; } = null!;
		public ApplicationUser CreatedBy { get; set; } = null!;
	}
}