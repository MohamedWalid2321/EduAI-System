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
        public async Task<IActionResult> AttemptQuiz(string quizCode)
        {
            var userId = User.GetUserId();
            var result = await _serviceManager.QuizAttemptService.StartQuizAsync(quizCode, userId);
            return Ok(result);
        }
        [HttpPost("submit/{attemptId}")]
        public async Task<IActionResult> SubmitQuiz(int attemptId, [FromBody] SubmitQuizRequestDto request)
        {
            var userId = User.GetUserId();
            var result = await _serviceManager.QuizAttemptService.SubmitQuizAsync(attemptId, request, userId);
            return Ok(result);
        }
        [HttpGet("result/{attemptId}")]
        public async Task<IActionResult> GetQuizResult(int attemptId)
        {
            var userId = User.GetUserId();
            var result = await _serviceManager.QuizAttemptService.GetQuizResultAsync(attemptId, userId);
            return Ok(result);
        }
        [HttpGet("quizzes/{quizCode}/students")]
        public async Task<IActionResult> GetStudentsByQuiz(string quizCode)
        {
             var result = await _serviceManager.QuizAttemptService.GetStudentsByQuizAsync(quizCode);
             return Ok(result);
        }
        [HttpGet("student/quizzes")]
        public async Task<IActionResult> GetStudentQuizzes()
        {
            var userId = User.GetUserId();
            var result = await _serviceManager.QuizAttemptService.GetStudentQuizzesAsync(userId);
            return Ok(result);
        }
    }
}
