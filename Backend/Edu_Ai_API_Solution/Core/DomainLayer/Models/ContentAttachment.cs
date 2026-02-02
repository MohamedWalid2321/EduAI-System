using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class ContentAttachment: BaseEntity<Guid>
	{
		public ContentAttachment()
		{
			Id = Guid.NewGuid();
		}
		public string FileName { get; set; } = null!; // title of the file
		public string FileUrl { get; set; } = null!; 

		public string ContentType { get; set; } = null!; // e.g., "application/pdf", "image/png"
		// Content RelationShip
		public int ContentId { get; set; }
		public Content Content { get; set; }
	}
}
