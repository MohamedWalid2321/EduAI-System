using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.ContentDto.ContentResponse
{
	public class ContentResponseDto
	{
		public int Id { get; set; }
		public string Title { get; set; } = null!;
		public string Body { get; set; } = null!;
		public ICollection<ContentAttachmentDto>? ContentAttachments { get; set; }
	}
}
