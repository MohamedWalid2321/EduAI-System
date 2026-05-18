using Shared.Dtos.QuizDto.Request;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace PresentationLayer.Controllers
{
    public class QuizController(IServiceManager serviceManager, ICacheService cacheService) : ApiControllerBase
    {
        private const string QuizzesPattern = "/api/quiz*";

        [HttpGet("course/{courseId}")]
        [HasPermission(Permissions.GetQuizzes)]
        [Cache(300)]
        public async Task<IActionResult> GetAllQuizzesByCourseId(int courseId, CancellationToken cancellationToken)
        {
            var quizzes = await serviceManager.QuizService.GetAllQuizzesForCourse(courseId, cancellationToken);
            return Ok(quizzes);
        }

        [HttpGet("{id}")]
        [HasPermission(Permissions.GetQuestions)]
        public async Task<IActionResult> GetQuizById(int id, CancellationToken cancellationToken)
        {
            var quiz = await serviceManager.QuizService.GetQuizByIdAsync(id, cancellationToken);
            return Ok(quiz);
        }

        [HttpPost("course/{courseId}")]
        [HasPermission(Permissions.AddOrUpdateQuiz)]
		public async Task<IActionResult> CreateOrUpdateQuizForCourse(int courseId, [FromBody] QuizRequestDto quizDto, CancellationToken cancellationToken)
        {
            var createdOrUpdatedQuiz = await serviceManager.QuizService.CreateOrUpdateQuizAsync(courseId, quizDto, cancellationToken);
            await cacheService.RemoveByPatternAsync(QuizzesPattern);
            return Ok(createdOrUpdatedQuiz);
        }

        [HttpDelete("{id}")]
        [HasPermission(Permissions.DeleteQuiz)]
		public async Task<IActionResult> DeleteQuiz(int id, CancellationToken cancellationToken)
        {
            await serviceManager.QuizService.DeleteQuizAsync(id, cancellationToken);
            await cacheService.RemoveByPatternAsync(QuizzesPattern);
            return Ok("Done");
        }
    }
}
