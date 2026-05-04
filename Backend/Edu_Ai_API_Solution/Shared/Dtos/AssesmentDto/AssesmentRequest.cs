using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AssesmentDto
{
	public class AssesmentRequest
	{
		[Range(0, 5)]
		public int AssType { get; set; }
		public double PercentageWeight { get; set; }
		public bool IsMandatory { get; set; }  
		public int Hours { get; set; }
	}
}
