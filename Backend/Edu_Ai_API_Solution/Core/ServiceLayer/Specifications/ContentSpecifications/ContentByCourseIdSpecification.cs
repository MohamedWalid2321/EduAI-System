using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.ContentSpecifications
{
	public class ContentByCourseIdSpecification: BaseSpecification<Content,int>
	{
		public ContentByCourseIdSpecification(int courseId) : base(c => c.CourseId == courseId)
		{
			AddInclude(p => p.ContentAttachments);
		}
	}
}
