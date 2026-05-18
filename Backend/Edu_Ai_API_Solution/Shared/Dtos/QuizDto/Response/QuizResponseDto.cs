using Shared.Dtos.QuizDto.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.QuizDto.Response
{
	public class QuizResponseDto
	{
        public int Id { get; set; }
        public string Title { get; set; }
		public DateTime ScheduledDate { get; set; }
		public TimeSpan Duration { get; set; } // duration of the quiz

		public List<QuestionResponseDto> QuizQuestions { get; set; } = [];
        public string QuizCode { get; set; } = null!; // a unique code for the quiz that students can use to access it (New) ##
        
    }
}
