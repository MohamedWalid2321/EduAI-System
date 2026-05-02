using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace ServiceAbstractionLayer
{
    public interface IQuestionService
    {
        Task<QuestionResponseDto> CreateQuestionForQuiz(int quizId, QuestionRequestDto questionRequest, CancellationToken cancellationToken = default);
        Task<QuestionResponseDto> UpdateQuestionAsync(int quizId, QuestionRequestDto questionRequest, CancellationToken cancellationToken = default);
        Task<QuestionResponseDto> ToggleQuestionAsync(int quizId, int questionId, CancellationToken cancellationToken = default);
    }
}
