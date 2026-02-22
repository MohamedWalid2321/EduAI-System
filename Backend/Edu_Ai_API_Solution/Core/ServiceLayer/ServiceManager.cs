using Microsoft.AspNetCore.Identity.UI.Services;
using Microsoft.Extensions.Logging;

namespace ServiceLayer
{
	public class ServiceManager(IUnitOfWork _unitOfWork,
		IFileStorageService _fileStorageService ,
		UserManager<ApplicationUser> _userManager,
		SignInManager<ApplicationUser> _signInManager,
		IJwtProvider _jwtProvider,
		IEmailSender _emailSender,
		IHttpContextAccessor _httpContextAccessor,
		IEmailBodyBuilder _emailBodyBuilder,
		ILogger<AuthService> _AuthLogger) : IServiceManager // <-- FIXED TYPE HERE
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

		private readonly Lazy<IAuthunticationService> _authunticationService =
			new Lazy<IAuthunticationService>(() => new Services.AuthService(_userManager, _signInManager, _jwtProvider, _fileStorageService, _httpContextAccessor,_emailSender, _emailBodyBuilder,_AuthLogger));
		public IAuthunticationService AuthunticationService => _authunticationService.Value;
		private readonly Lazy<IUserService> _userService =
			new Lazy<IUserService>(() => new Services.UserService(_userManager));
		public IUserService UserService => _userService.Value;
	}
}
