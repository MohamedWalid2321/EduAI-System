using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.QuizDto
{
	public class QuestionChoicesDto
	{
		public int Id { get; set; }
		public string ChoiceText { get; set; }
		public bool IsCorrect { get; set; }
	}
}
