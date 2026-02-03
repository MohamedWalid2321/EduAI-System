using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.QuizDto.Request
{
	public class QuizRequestDto
	{
		public int? Id { get; set; }
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public DateTime ScheduledDate { get; set; } // when the quiz is scheduled to take place (New) ##
		public TimeSpan Duration { get; set; } // duration of the quiz
		public double TotalMarks { get; set; }
	}
}
