using Shared.Dtos.QuizDto.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PresentationLayer.Controllers
{
    public class QuestionController(IServiceManager _serviceManager) : ApiControllerBase
    {
        [HttpPost("/Quiz/{quizId}")]
        public async Task<IActionResult> CreateQuestionsForQuiz(int quizId, [FromBody] QuestionRequestDto questionRequest)
        {
            var questions = await _serviceManager.QuestionService.CreateQuestionForQuiz(quizId, questionRequest);
            return Ok(questions);
        }
        [HttpPost("/ToggleStatus/QuizId/{quizId}/QuestionId{questionId}")]
        public async Task<IActionResult> ToggleQuestionStatus(int quizId, int questionId)
        {
            var question = await _serviceManager.QuestionService.ToggleQuestionAsync(quizId, questionId);
            return Ok(question);
        }
        [HttpPost("/Update/QuizId/{quizId}")]
        public async Task<IActionResult> UpdateQuestionStatus(int quizId, [FromBody] QuestionRequestDto questionRequest)
        {
            var question = await _serviceManager.QuestionService.UpdateQuestionAsync(quizId, questionRequest);
            return Ok(question);
        }

    }
}
