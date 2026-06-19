using Shared.Dtos.QuizDto.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.QuizDto.Response
{
	public class QuizResponseInDetailsDto
	{
		public int Id { get; set; }
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public DateTime ScheduledDate { get; set; }
		public TimeSpan Duration { get; set; }       // duration of the quiz
		public double TotalMarks { get; set; }        // total marks for the quiz
		public bool IsActive { get; set; }            // indicates if the quiz is currently active
		public string QuizCode { get; set; } = null!; // unique code students use to access the quiz
		public int CourseId { get; set; }             // owning course

		public List<QuestionResponseDto> QuizQuestions { get; set; } = [];
	}
}
