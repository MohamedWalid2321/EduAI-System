using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.ContentDto.ContentRequest
{
	public class ContentRequestDto
	{
		public int Id { get; set; }
		public string Title { get; set; } = null!;
		public string Body { get; set; } = null!;

	}
}
