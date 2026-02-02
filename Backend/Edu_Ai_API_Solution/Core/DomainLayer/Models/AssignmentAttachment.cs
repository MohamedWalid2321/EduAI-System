using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class AssignmentAttachment : BaseEntity<Guid>
	{
		public AssignmentAttachment()
		{
			Id = Guid.NewGuid();
		}
		public string FileName { get; set; } = null!; // title of the file
		public string FileUrl { get; set; } = null!;
		public string Type { get; set; } = null!; // e.g., "application/pdf", "image/png" 
		// Assignment RelationShip
		public int AssignmentId { get; set; }
		public Assignment Assignment { get; set; }
	}
}
