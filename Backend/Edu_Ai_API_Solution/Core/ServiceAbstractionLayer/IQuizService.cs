using Shared.Dtos.QuizDto;
using Shared.Dtos.QuizDto.Request;
using Shared.Dtos.QuizDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
	public interface IQuizService
	{
		Task<QuizResponseDto> CreateOrUpdateQuizAync(int CourseId,QuizRequestDto quizRequest);
		Task<QuizResponseDto> GetQuizByIdAsync(int quizId);
		Task<IEnumerable<QuizResponseDto>> GetAllQuizzesAsync();
		Task<IEnumerable<QuizResponseDto>> GetAllQuizzesForCourse(int CourseId);
		Task DeleteQuizAsync(int quizId);
		Task<QuizResponseDto> AddQuestionToQuiz(int QuizId, ICollection<QuizQuestionDto> Questions);
		
	}
}
