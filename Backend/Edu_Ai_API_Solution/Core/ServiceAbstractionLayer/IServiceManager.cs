namespace ServiceAbstractionLayer
{
	public interface IServiceManager
	{
		IDepartmentService DepartmentService { get; }
		ICourseService CourseService { get; }
		IContentService ContentService { get; }
		IAssigmentService AssignmentService { get; }
		IAuthunticationService AuthunticationService { get; }
		IUserService UserService { get; }
		IRoleService RoleService { get; }
		IQuizService QuizService { get; }
		IQuestionService QuestionService { get; }
		IQuizAttemptService QuizAttemptService { get; }
		IAssignmentSubmissionService AssignmentSubmissionService { get; }
		IAcademicYearService AcademicYearService { get; }
		IFeesService FeesService { get; }

		IPaymentService PaymentService { get; }

        IPaymobService PaymentGateway { get; }


    }
}
