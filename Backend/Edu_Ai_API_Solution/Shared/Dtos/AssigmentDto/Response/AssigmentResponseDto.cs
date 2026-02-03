using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AssigmentDto.Response
{
	public class AssigmentResponseDto
	{
		public int Id { get; set; } 
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public DateTime DueDate { get; set; }
		public double TotalMarks { get; set; }
		public ICollection<AssigmentAttachmentDto>? AssignmentAttachments { get; set; }
	}
}
