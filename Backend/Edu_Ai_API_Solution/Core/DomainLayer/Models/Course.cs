using DomainLayer.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class Course:BaseEntity<int>
	{
		public string Title { get; set; } = null!;
		public string Description { get; set; } = null!;
		public Semster semster { get; set; }
		public string ImageUrl { get; set; } = null!;
		public int Credit_Hour { get; set; }
		public CourseStatus CourseStatus { get; set; }
		public String LearningOutcomes { get; set; } = null!;
		// Self RelationShip
		public int? PrerequisiteCourseId { get; set; }
		public Course PrerequisiteCourse { get; set; }
		// Department RelationShip
		public ICollection<Department> Departments { get; set; }
		// Assessment RelationShip
		public ICollection<Assessment> Assessments { get; set; }
		// Content RelationShip
		public ICollection<Content> Contents { get; set; }
		// Assignment RelationShip
		public ICollection<Assignment> Assignments { get; set; }
		// Quiz RelationShip
		public ICollection<Quiz> Quizzes { get; set; }




	}
}
