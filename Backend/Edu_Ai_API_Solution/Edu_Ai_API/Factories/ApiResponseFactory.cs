using Microsoft.AspNetCore.Mvc;
using Shared.ErrorModels;

namespace Edu_Ai_API.Factories
{
    public static class ApiResponseFactory
    {
        public static IActionResult GenerateApiValidationResponse(ActionContext context)
        {
            var errors = context.ModelState
                        .Where(e => e.Value.Errors.Any())
                        .Select(m => new ValidationError()
                        {
                            Field = m.Key,
                            Errors = m.Value.Errors.Select(er => er.ErrorMessage)
                        });
            var response = new ValidationErrorToReturn
            {
                validationErrors = errors
            };
            return new BadRequestObjectResult(response);
        }
    }
}
