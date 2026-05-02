using Shared.Dtos.AttemptQuiz.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
    public class QuizAttemptsController(IServiceManager _serviceManager) : ApiControllerBase
    {
        [HttpGet("attempt/{quizCode}")]
        public async Task<IActionResult> AttemptQuiz(string quizCode, CancellationToken cancellationToken)
        {
            var userId = User.GetUserId();
            var result = await _serviceManager.QuizAttemptService.StartQuizAsync(quizCode, userId, cancellationToken);
            return Ok(result);
        }

        [HttpPost("submit/{attemptId}")]
        public async Task<IActionResult> SubmitQuiz(int attemptId, [FromBody] SubmitQuizRequestDto request, CancellationToken cancellationToken)
        {
            var userId = User.GetUserId();
            var result = await _serviceManager.QuizAttemptService.SubmitQuizAsync(attemptId, request, userId, cancellationToken);
            return Ok(result);
        }

        [HttpGet("result/{attemptId}")]
        public async Task<IActionResult> GetQuizResult(int attemptId, CancellationToken cancellationToken)
        {
            var userId = User.GetUserId();
            var result = await _serviceManager.QuizAttemptService.GetQuizResultAsync(attemptId, userId, cancellationToken);
            return Ok(result);
        }

        [HttpGet("quizzes/{quizCode}/students")]
        public async Task<IActionResult> GetStudentsByQuiz(string quizCode, CancellationToken cancellationToken)
        {
            var result = await _serviceManager.QuizAttemptService.GetStudentsByQuizAsync(quizCode, cancellationToken);
            return Ok(result);
        }

        [HttpGet("student/quizzes")]
        public async Task<IActionResult> GetStudentQuizzes(CancellationToken cancellationToken)
        {
            var userId = User.GetUserId();
            var result = await _serviceManager.QuizAttemptService.GetStudentQuizzesAsync(userId, cancellationToken);
            return Ok(result);
        }
    }
}
