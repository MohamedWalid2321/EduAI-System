using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.QuizDto.Request
{
	public class QuestionRequestDto
	{
		public int Id { get; set; }
		public string QuestionText { get; set; } = null!; // the text of the question or heading (Header)
		public string QuestionType { get; set; }
		public double Marks { get; set; } // marks or points allocated for the question
		//public ICollection<QuestionChoicesDto> QuestionChoices { get; set; }
		public int CorrectAnswerIndex { get; set; } // index of the correct answer in the Answers list

        public List<string> QuestionChoices { get; set; } = [];
    }
}
