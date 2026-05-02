using Shared.Dtos.QuizDto.Request;

namespace PresentationLayer.Controllers
{
    public class QuizController(IServiceManager serviceManager) : ApiControllerBase
    {
        [HttpGet("course/{courseId}")]
        public async Task<IActionResult> GetAllQuizzesByCourseId(int courseId, CancellationToken cancellationToken)
        {
            var quizzes = await serviceManager.QuizService.GetAllQuizzesForCourse(courseId, cancellationToken);
            return Ok(quizzes);
        }

        [HttpPost("course/{courseId}")]
        public async Task<IActionResult> CreateOrUpdateQuizForCourse(int courseId, [FromBody] QuizRequestDto quizDto, CancellationToken cancellationToken)
        {
            var createdOrUpdatedQuiz = await serviceManager.QuizService.CreateOrUpdateQuizAsync(courseId, quizDto, cancellationToken);
            return Ok(createdOrUpdatedQuiz);
        }

        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteQuiz(int id, CancellationToken cancellationToken)
        {
            await serviceManager.QuizService.DeleteQuizAsync(id, cancellationToken);
            return Ok("Done");
        }

        [HttpGet("{id}")]
        public async Task<IActionResult> GetQuizById(int id, CancellationToken cancellationToken)
        {
            var quiz = await serviceManager.QuizService.GetQuizByIdAsync(id, cancellationToken);
            return Ok(quiz);
        }
    }
}
