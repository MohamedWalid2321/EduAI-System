using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.ContentDto
{
	public class ContentAttachmentDto
	{
		public string FileName { get; set; } = null!; // title of the file
		public string FileUrl { get; set; } = null!;

		public string ContentType { get; set; } = null!; // e.g., "application/pdf", "image/png"
														
	}
}
