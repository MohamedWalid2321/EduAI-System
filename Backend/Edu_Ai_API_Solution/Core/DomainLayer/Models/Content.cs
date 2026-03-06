using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class Content: BaseEntity<int>
	{
		public string Title { get; set; } = null!;
		public string Body { get; set; } = null!;

		// Course RelationShip
		public int CourseId { get; set; }
		public Course Course { get; set; }= null!;

		// ContentAttachment RelationShip
		public ICollection<ContentAttachment> ContentAttachments { get; set; } = [];


	}
}
