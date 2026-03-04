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
		public bool IsActive { get; set; } = true; // indicates if the question is currently active or not
        // Quiz RelationShip
        public int QuizId { get; set; }
		public Quiz Quiz { get; set; }
		// question Choices relation
		public ICollection<QuestionChoices> QuestionChoices { get; set; } = [];


	}
}
