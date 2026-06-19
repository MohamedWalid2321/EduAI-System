namespace ServiceAbstractionLayer
{
	public interface IQuizService
	{
		Task<QuizResponseDto> CreateOrUpdateQuizAsync(int CourseId, QuizRequestDto quizRequest, CancellationToken cancellationToken = default);
        Task<QuizResponseInDetailsDto> GetQuizByIdAsync(int quizId, CancellationToken cancellationToken = default);
		Task<IEnumerable<QuizResponseDto>> GetAllQuizzesAsync(CancellationToken cancellationToken = default);
		Task<IEnumerable<QuizResponseDto>> GetAllQuizzesForCourse(int CourseId, bool onlyActive, CancellationToken cancellationToken = default);
		Task DeleteQuizAsync(int quizId, CancellationToken cancellationToken = default);
		Task<bool> ToggleQuizActiveAsync(int quizId, CancellationToken cancellationToken = default);
	}
}
