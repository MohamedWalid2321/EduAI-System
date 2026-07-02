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

     
        [HttpGet("quizzes/{quizId:int}/attempts")]
		[HasPermission(Permissions.AddOrUpdateQuiz)]
		public async Task<IActionResult> GetAttemptsByQuiz(int quizId, CancellationToken cancellationToken)
        {
            var result = await _serviceManager.QuizAttemptService.GetAttemptsByQuizIdAsync(quizId, cancellationToken);
            return Ok(result);
        }

        
        [HttpGet("attempts/{attemptId:int}/details")]
		[HasPermission(Permissions.AddOrUpdateQuiz)]
		public async Task<IActionResult> GetAttemptDetails(int attemptId, CancellationToken cancellationToken)
        {
            var result = await _serviceManager.QuizAttemptService.GetAttemptDetailsByIdAsync(attemptId, cancellationToken);
            return Ok(result);
        }
        [HttpPatch("attempts/{attemptId:int}/score/finalize")]
        [HasPermission(Permissions.FinalizeAttemptScore)]
        public async Task<IActionResult> FinalizeAttemptScore(
            int attemptId,
            [FromBody] UpdateAttemptScoreRequestDto request,
            CancellationToken cancellationToken)
        {
            var result = await _serviceManager.QuizAttemptService
                .FinalizeAttemptScoreAsync(attemptId, request.Score, cancellationToken);
            return Ok(result);
        }
        [HttpPatch("attempts/{attemptId:int}/score")]
        [HasPermission(Permissions.UpdateAttemptScore)]
        public async Task<IActionResult> UpdateAttemptScore(
            int attemptId,
            [FromBody] UpdateAttemptScoreRequestDto request,
            CancellationToken cancellationToken)
        {
            var result = await _serviceManager.QuizAttemptService
                .UpdateAttemptScoreAsync(attemptId, request.Score, cancellationToken);
            return Ok(result);
        }

        [HttpGet("courses/{courseId:int}/students/{userId}/grades")]
        [HasPermission(Permissions.GetStudentCourseGrades)]
        public async Task<IActionResult> GetStudentCourseGrades(
            int courseId,
            string userId,
            CancellationToken cancellationToken)
        {
            var result = await _serviceManager.QuizAttemptService
                .GetStudentGradesByCourseAsync(courseId, userId, cancellationToken);
            return Ok(result);
        }
    }
}
