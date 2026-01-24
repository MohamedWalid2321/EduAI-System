using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class Assignment: BaseEntity<int>
	{
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public DateTime DueDate { get; set; } // deadline for submission
		public double TotalMarks { get; set; } // points or marks for the assignment
		// Course RelationShip
		public int CourseId { get; set; }
		public Course Course { get; set; }
		// AssignmentAttachment RelationShip
		public ICollection<AssignmentAttachment> AssignmentAttachments { get; set; }

	}
}
