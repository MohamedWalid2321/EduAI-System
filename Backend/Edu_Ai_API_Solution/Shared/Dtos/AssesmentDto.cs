using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos
{
	public class AssesmentDto
	{
		public int AssType { get; set; }
		public double PercentageWeight { get; set; }
		public bool IsMandatory { get; set; }  
		public int Hours { get; set; }
	}
}
