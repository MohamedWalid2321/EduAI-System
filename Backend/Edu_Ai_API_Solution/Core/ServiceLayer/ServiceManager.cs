using DomainLayer.Contracts;
using DomainLayer.Models;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Identity.UI.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using ServiceAbstractionLayer;
using ServiceLayer.Services;

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
        IPaymobService _paymentGateway,
		IEnrollmentService _enrollmentService,
		IConfiguration _configuration,
		ILogger<AuthService> _authLogger,
		IBackgroundJobClient _backgroundJobClient) : IServiceManager
    {
		#region Department
		private readonly Lazy<IDepartmentService> _departmentService =
            new Lazy<IDepartmentService>(() => new Services.DepartmentService(_unitOfWork));
		public IDepartmentService DepartmentService => _departmentService.Value;
		#endregion

		#region Course
		private readonly Lazy<ICourseService> _courseService =
			new Lazy<ICourseService>(() => new Services.CourseService(_unitOfWork, _fileStorageService, _userManager, _roleManager));
		public ICourseService CourseService => _courseService.Value;
		#endregion

		#region Content
		private readonly Lazy<IContentService> _contentService =
			new Lazy<IContentService>(() => new Services.ContentService(_unitOfWork, _fileStorageService));
		public IContentService ContentService => _contentService.Value;
		#endregion

		#region Assignment
		private readonly Lazy<IAssigmentService> _assignmentService =
			new Lazy<IAssigmentService>(() => new Services.AssignmentService(_unitOfWork, _fileStorageService));
		public IAssigmentService AssignmentService => _assignmentService.Value;
		#endregion

		#region Quiz
		private readonly Lazy<IQuizService> _quizService =
			new Lazy<IQuizService>(() => new Services.QuizService(_unitOfWork));
		public IQuizService QuizService => _quizService.Value;
		#endregion

		private readonly Lazy<IQuestionService> _questionService =
			new Lazy<IQuestionService>(() => new Services.QuestionService(_unitOfWork));
		public IQuestionService QuestionService => _questionService.Value;

		private readonly Lazy<IQuizAttemptService> quizAttemptService =
			new Lazy<IQuizAttemptService>(() => new Services.QuizAttemptService(_unitOfWork));
		public IQuizAttemptService QuizAttemptService => quizAttemptService.Value;

		private readonly Lazy<IAssignmentSubmissionService> assignmentSubmissionService =
			new Lazy<IAssignmentSubmissionService>(() => new Services.AssignmentSubmissionService(_unitOfWork, _fileStorageService));
		public IAssignmentSubmissionService AssignmentSubmissionService => assignmentSubmissionService.Value;

        private readonly Lazy<IAcademicYearService> academicYearService =
            new Lazy<IAcademicYearService>(() => new Services.AcademicYearService(_unitOfWork));
        public IAcademicYearService AcademicYearService => academicYearService.Value;

        private readonly Lazy<IFeesService> feesService =
            new Lazy<IFeesService>(() => new Services.FeesService(_unitOfWork));
        public IFeesService FeesService => feesService.Value;

        public IPaymobService PaymentGateway => _paymentGateway;

        private readonly Lazy<IPaymentService> paymentService =
            new Lazy<IPaymentService>(() => new Services.PaymentService(_unitOfWork, _userManager, _paymentGateway));
        public IPaymentService PaymentService => paymentService.Value;

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
				_authLogger,
				_backgroundJobClient));
		public IAuthunticationService AuthunticationService => _authunticationService.Value;

        public IRoleService RoleService => _roleService;
        public IUserService UserService => _userService;

		private readonly Lazy<ILectureService> _lectureService =
			new Lazy<ILectureService>(() => new Services.LectureService(_unitOfWork, _configuration, _userManager));
		public ILectureService LectureService => _lectureService.Value;

		private readonly Lazy<IEnrollmentService> _lazyEnrollmentService =
			new Lazy<IEnrollmentService>(() => new Services.EnrollmentService(_unitOfWork, _userManager));
		public IEnrollmentService EnrollmentService => _lazyEnrollmentService.Value;

		#region Notification
		private readonly Lazy<INotificationService> _notificationService =
			new Lazy<INotificationService>(() => new Services.NotificationService(_unitOfWork));
		public INotificationService NotificationService => _notificationService.Value;
		#endregion
	}
}
