using Microsoft.AspNetCore.Identity.UI.Services;
using Microsoft.Extensions.Logging;
using ServiceAbstractionLayer;
using DomainLayer.Contracts;
using DomainLayer.Models;
using Microsoft.AspNetCore.Identity;


namespace ServiceLayer
{
    public class ServiceManager(
        IUnitOfWork _unitOfWork,
        IFileStorageService _fileStorageService,
        UserManager<ApplicationUser> _userManager,
        SignInManager<ApplicationUser> _signInManager,
        RoleManager<ApplicationRole> _roleManager,
        IJwtProvider _jwtProvider,
        IEmailSender _emailSender,
        IHttpContextAccessor _httpContextAccessor,
        IEmailBodyBuilder _emailBodyBuilder,
        IRoleService _roleService,  
        IUserService _userService,   
        ILogger<AuthService> _authLogger) : IServiceManager
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

        private readonly Lazy<IQuizService> _quizService =
            new Lazy<IQuizService>(() => new Services.QuizService(_unitOfWork));
        public IQuizService QuizService => _quizService.Value;

        private readonly Lazy<IQuestionService> _questionService =
            new Lazy<IQuestionService>(() => new Services.QuestionService(_unitOfWork));
        public IQuestionService QuestionService => _questionService.Value;

        private readonly Lazy<IQuizAttemptService> quizAttemptService =
            new Lazy<IQuizAttemptService>(() => new Services.QuizAttemptService(_unitOfWork));
        public IQuizAttemptService QuizAttemptService => quizAttemptService.Value;

        private readonly Lazy<IAuthunticationService> _authunticationService =
            new Lazy<IAuthunticationService>(() => new Services.AuthService(
                _userManager, 
                _signInManager, 
                _jwtProvider, 
                _fileStorageService, 
                _httpContextAccessor,
                _emailSender, 
                _emailBodyBuilder,
                _roleManager,
                _unitOfWork,
				_authLogger));
        public IAuthunticationService AuthunticationService => _authunticationService.Value;

        public IRoleService RoleService => _roleService;
        public IUserService UserService => _userService;
    }
}
