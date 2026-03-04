namespace ServiceAbstractionLayer
{
	public interface IQuizService
	{
		Task<QuizResponseDto> CreateOrUpdateQuizAsync(int CourseId,QuizRequestDto quizRequest);
        Task<QuizResponseDto> GetQuizByIdAsync(int quizId);
		Task<IEnumerable<QuizResponseDto>> GetAllQuizzesAsync();
		Task<IEnumerable<QuizResponseDto>> GetAllQuizzesForCourse(int CourseId);
		Task DeleteQuizAsync(int quizId);
		
		
	}
}
