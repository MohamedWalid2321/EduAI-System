using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.ContentSpecifications
{
	public class ContentSpecification:BaseSpecification<Content,int>
	{
		public ContentSpecification(int id) : base(p => p.Id == id)
		{
			AddInclude(p => p.ContentAttachments);
			AddInclude(p => p.Course);
		}
		public ContentSpecification() : base(null)
		{
			AddInclude(p => p.ContentAttachments);
		}
	}
}
