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
		// Remmark : I want to add property Status to know if the assigment is late or still deadline not reached make it after handling the submission feature
		// Course RelationShip
		public int CourseId { get; set; }
		public Course Course { get; set; }
		// AssignmentAttachment RelationShip
		public ICollection<AssignmentAttachment> AssignmentAttachments { get; set; }

	}
}
