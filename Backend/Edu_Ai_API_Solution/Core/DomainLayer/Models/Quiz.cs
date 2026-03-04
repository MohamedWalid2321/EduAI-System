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
		public bool IsActive { get; set; } = true; // indicates if the quiz is currently active or not
		public string QuizCode { get; set; } = null!; // a unique code for the quiz that students can use to access it (New) ##
   
        public int CourseId { get; set; }// Course RelationShip
        public Course Course { get; set; }
		
		public ICollection<QuizQuestion> QuizQuestions { get; set; }= [];// QuizQuestion RelationShip
    }
}
