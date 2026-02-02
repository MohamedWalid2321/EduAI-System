using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.ContentDto
{
	public class ContentAttachmentDto
	{
		public Guid Id { get; set; }
		public string FileName { get; set; } = null!;
		public string FileUrl { get; set; } = null!;

		public string ContentType { get; set; } = null!; 
														
	}
}
