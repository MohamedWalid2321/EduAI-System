using Shared.Dtos.AttemptQuiz;
using Shared.Dtos.AttemptQuiz.Request;
using Shared.Dtos.AttemptQuiz.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IQuizAttemptService
    {
        Task<StartQuizResponseDto> StartQuizAsync(string QuizCode, string studentId);

        Task<SubmitQuizResponseDto> SubmitQuizAsync(int attemptId, SubmitQuizRequestDto request, string studentId);

        Task<SubmitQuizResponseDto> GetQuizResultAsync(int attemptId, string studentId);

        
        Task<List<StudentAttemptDto>> GetStudentsByQuizAsync(string quizCode);

       
        Task<List<StudentQuizDto>> GetStudentQuizzesAsync(string studentId);
    }
}
