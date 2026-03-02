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
<<<<<<< HEAD
=======
        public ICollection<ApplicationUser>? Users { get; set; }
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
	}
}
