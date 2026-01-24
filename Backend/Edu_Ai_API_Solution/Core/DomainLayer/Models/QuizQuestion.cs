using DomainLayer.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class QuizQuestion : BaseEntity<int>
	{
		public string QuestionText { get; set; } = null!; // the text of the question or heading (Header)
		public QuestionTypes QuestionType { get; set; } 
		public double Marks { get; set; } // marks or points allocated for the question
		// Quiz RelationShip
		public int QuizId { get; set; }
		public Quiz Quiz { get; set; }
		// question Choices relation
		public ICollection<QuestionChoices> QuestionChoices { get; set; }


	}
}
