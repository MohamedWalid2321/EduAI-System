using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public class BaseEntity<Tkey>
	{
		public Tkey Id { get; set; } //pk
		public DateTime? CreatedAt { get; set; } = DateTime.Now;
		public DateTime? LastUpdatedAt { get; set; }
		public string? CreatedBy { get; set; }
		public string? LastUpdatedBy { get; set;}

	}
}
