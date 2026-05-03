using ServiceAbstractionLayer;

namespace ServiceAbstractionLayer
{
	public interface IServiceManager
	{
		IDepartmentService DepartmentService { get; }
		ICourseService CourseService { get; }
		IContentService ContentService { get; }
		IAssigmentService AssignmentService { get; }
		IQuizService QuizService { get; }
		IQuestionService QuestionService { get; }
		IQuizAttemptService QuizAttemptService { get; }
		IAssignmentSubmissionService AssignmentSubmissionService { get; }
		IAcademicYearService AcademicYearService { get; }
		IUserService UserService { get; }
		IRoleService RoleService { get; }
		IFeesService FeesService { get; }
		IPaymentService PaymentService { get; }
		IPaymobService PaymentGateway { get; }
		IAuthunticationService AuthunticationService { get; }
		ILectureService LectureService { get; }
		INotificationService NotificationService { get; }
	}
}
