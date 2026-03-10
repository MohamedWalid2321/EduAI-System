using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class QuestionChoices: BaseEntity<int>
	{
		public string ChoiceText { get; set; }
		public bool IsCorrect { get; set; }
		public int QuizQuestionId { get; set; } // QuizQuestion RelationShip
        public QuizQuestion QuizQuestion { get; set; }
	}
}
