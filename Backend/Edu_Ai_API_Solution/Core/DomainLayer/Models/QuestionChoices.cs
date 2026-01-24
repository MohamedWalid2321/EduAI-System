using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class QuestionChoices
	{
		public int Id { get; set; }
		public string ChoiceText { get; set; }
		public bool IsCorrect { get; set; }
		// QuizQuestion RelationShip
		public int QuizQuestionId { get; set; }
		public QuizQuestion QuizQuestion { get; set; }
	}
}
