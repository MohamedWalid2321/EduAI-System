using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
	public interface IServiceManager
	{
		IDepartmentService DepartmentService { get; }
		ICourseService CourseService { get; }
		IContentService ContentService { get; }
	}
}
