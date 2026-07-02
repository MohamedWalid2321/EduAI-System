using Shared.Dtos.AttemptQuiz;
using Shared.Dtos.AttemptQuiz.Request;
using Shared.Dtos.AttemptQuiz.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace ServiceAbstractionLayer
{
    public interface IQuizAttemptService
    {
        Task<StartQuizResponseDto> StartQuizAsync(string QuizCode, string studentId, CancellationToken cancellationToken = default);

        Task<SubmitQuizResponseDto> SubmitQuizAsync(int attemptId, SubmitQuizRequestDto request, string studentId, CancellationToken cancellationToken = default);

        Task<SubmitQuizResponseDto> GetQuizResultAsync(int attemptId, string studentId, CancellationToken cancellationToken = default);

        Task<List<StudentAttemptDto>> GetStudentsByQuizAsync(string quizCode, CancellationToken cancellationToken = default);

        Task<List<StudentQuizDto>> GetStudentQuizzesAsync(string studentId, CancellationToken cancellationToken = default);
        Task<List<QuizAttemptDetailsDto>> GetAttemptsByQuizIdAsync(int quizId, CancellationToken cancellationToken = default);
        Task<QuizAttemptDetailsDto> GetAttemptDetailsByIdAsync(int attemptId, CancellationToken cancellationToken = default);
        Task<QuizAttemptDetailsDto> FinalizeAttemptScoreAsync(int attemptId, int newScore, CancellationToken cancellationToken = default);
        Task<QuizAttemptDetailsDto> UpdateAttemptScoreAsync(int attemptId, int newScore, CancellationToken cancellationToken = default);
        Task<List<StudentCourseGradeDto>> GetStudentGradesByCourseAsync(int courseId, string studentId, CancellationToken cancellationToken = default);
    }
}
