using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class Department: BaseEntity<int>
	{
		public string Title { get; set; } = null!;
		public ICollection<Course>? courses { get; set; }
        public ICollection<ApplicationUser>? Users { get; set; }
	}
}
