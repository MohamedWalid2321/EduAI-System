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
		IAuthunticationService AuthunticationService { get; }
		IRoleService RoleService { get; }
		IUserService UserService { get; }
		IEnrollmentService EnrollmentService { get; }
		ILectureService LectureService { get; }
	}
}
