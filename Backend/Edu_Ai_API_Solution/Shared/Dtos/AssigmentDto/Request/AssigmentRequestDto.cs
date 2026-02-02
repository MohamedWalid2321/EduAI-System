using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AssigmentDto.Request
{
	public class AssigmentRequestDto
	{
		public int Id { get; set; } // for update scenarios
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public DateTime DueDate { get; set; } // deadline for submission
		public double TotalMarks { get; set; } // points or marks for the assignment
	}
}
