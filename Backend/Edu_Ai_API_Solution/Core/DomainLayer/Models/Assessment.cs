using DomainLayer.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class Assessment: BaseEntity<int>
	{
		public AssTypes AssType { get; set; }
		public double PercentageWeight { get; set; }
		public bool IsMandatory { get; set; } // why? to know if the assessment is mandatory or not
		public int Hours { get; set; }
		// Course RelationShip
		public int CourseId { get; set; }
		public Course Course { get; set; }


	}
}
