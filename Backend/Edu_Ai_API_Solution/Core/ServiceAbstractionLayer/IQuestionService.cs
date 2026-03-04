using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IQuestionService
    {
        Task<QuestionResponseDto> CreateQuestionForQuiz(int quizId, QuestionRequestDto questionRequest);
        Task<QuestionResponseDto> UpdateQuestionAsync(int quizId, QuestionRequestDto questionRequest);
        Task<QuestionResponseDto> ToggleQuestionAsync(int quizId, int questionId);


    }
}
