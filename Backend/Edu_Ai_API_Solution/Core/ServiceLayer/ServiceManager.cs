using DomainLayer.Contracts;
using ServiceAbstractionLayer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer
{
	public class ServiceManager(IUnitOfWork _unitOfWork,IFileStorageService _fileStorageService) : IServiceManager
	{
		private readonly Lazy<IDepartmentService> _departmentService =
			new Lazy<IDepartmentService>(() => new Services.DepartmentService(_unitOfWork));
		public IDepartmentService DepartmentService => _departmentService.Value;
		
		private readonly Lazy<ICourseService> _courseService =
			new Lazy<ICourseService>(() => new Services.CourseService(_unitOfWork, _fileStorageService));
		public ICourseService CourseService => _courseService.Value;

		private readonly Lazy<IContentService> _contentService =
			new Lazy<IContentService>(() => new Services.ContentService(_unitOfWork, _fileStorageService));
		public IContentService ContentService => _contentService.Value;

		private readonly Lazy<IAssigmentService> _assignmentService =
			new Lazy<IAssigmentService>(() => new Services.AssignmentService(_unitOfWork, _fileStorageService));
		public IAssigmentService AssignmentService => _assignmentService.Value;

		
    }
}
