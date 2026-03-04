using Shared.Dtos.QuizDto.Request;

namespace PresentationLayer.Controllers
{

    public class QuizController(IServiceManager serviceManager) : ApiControllerBase
    {
        [HttpGet("course/{courseId}")]
        public async Task<IActionResult> GetAllQuizzesByCourseId(int courseId)
        {
            var quizzes = await serviceManager.QuizService.GetAllQuizzesForCourse(courseId);
            return Ok(quizzes);
        }
        [HttpPost("course/{courseId}")]
        public async Task<IActionResult> CreateOrUpdateQuizForCourse(
            int courseId,
            [FromBody] QuizRequestDto quizDto)
        {
            var createdOrUpdatedQuiz = await serviceManager.QuizService
                .CreateOrUpdateQuizAsync(courseId, quizDto);
            return Ok(createdOrUpdatedQuiz);
        }
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteQuiz(int id)
        {
            await serviceManager.QuizService.DeleteQuizAsync(id);
            return Ok("Done");
        }
        [HttpGet("{id}")]
        public async Task<IActionResult> GetQuizById(int id)
        {
            var quiz = await serviceManager.QuizService.GetQuizByIdAsync(id);
            return Ok(quiz);
        }
    }
}
