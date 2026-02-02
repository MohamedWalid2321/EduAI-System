using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AssigmentDto
{
	public class AssigmentAttachmentDto
	{
		public Guid Id { get; set; } // Added - needed to identify attachment
		public string FileName { get; set; } = null!;
		public string FileUrl { get; set; } = null!;
		public string Type { get; set; } = null!;
	}
}
