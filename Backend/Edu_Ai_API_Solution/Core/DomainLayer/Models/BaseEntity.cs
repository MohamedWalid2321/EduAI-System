using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
	public abstract class BaseEntity
	{
		public DateTime? CreatedAt { get; set; } = DateTime.Now;
		public DateTime? LastUpdatedAt { get; set; }
		public string? CreatedBy { get; set; }
		public string? LastUpdatedBy { get; set; }
		public bool IsDeleted { get; set; } = false;       
		public DateTime? DeletedAt { get; set; }
		public string? DeletedBy { get; set; }
	}

	public class BaseEntity<Tkey> : BaseEntity
	{
		public Tkey Id { get; set; } //pk
	}
}
