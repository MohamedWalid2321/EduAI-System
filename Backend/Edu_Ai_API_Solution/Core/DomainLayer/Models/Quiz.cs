using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class Quiz: BaseEntity<int>
	{
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public DateTime ScheduledDate { get; set; } // when the quiz is scheduled to take place (New) ##
		public TimeSpan Duration { get; set; } // duration of the quiz
		public double TotalMarks { get; set; } // total marks for the quiz
		// Remark : I Want to add property IsActive to know if the quiz is currently active or not after handling the scheduling feature and the taking quiz feature
		// Course RelationShip
		public int CourseId { get; set; }
		public Course Course { get; set; }
		// QuizQuestion RelationShip
		public ICollection<QuizQuestion> QuizQuestions { get; set; }
	}
}
