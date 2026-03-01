using DomainLayer.Exceptions;
using Shared.ErrorModels;

namespace Edu_Ai_API.CustomMiddleWares
{
    public class CustomExceptionHandlerMiddleWare
    {
        private readonly RequestDelegate _next;
        private readonly ILogger _logger;
        public CustomExceptionHandlerMiddleWare(RequestDelegate Next, ILogger<CustomExceptionHandlerMiddleWare> logger)
        {
            _next = Next;
            _logger = logger;
        }

        public async Task InvokeAsync(HttpContext httpContext)
        {
            try
            {
                await _next.Invoke(httpContext);
                
                // Only handle error responses if the response hasn't started writing
                if (!httpContext.Response.HasStarted)
                {
                    await HandleErrorResponseAsync(httpContext);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "An unhandled exception occurred while processing the request.");
                
                if (!httpContext.Response.HasStarted)
                {
                    await HandleExceptionAsync(httpContext, ex);
                }
            }
        }

        private static async Task HandleExceptionAsync(HttpContext httpContext, Exception ex)
        {
            httpContext.Response.StatusCode = ex switch
            {
                NotFoundException => StatusCodes.Status404NotFound,
                UnAuthorizedException => StatusCodes.Status401Unauthorized,
                ConflictException => StatusCodes.Status409Conflict,
                BadRequestException => StatusCodes.Status400BadRequest,
				_ => StatusCodes.Status500InternalServerError
            };
            httpContext.Response.ContentType = "application/json";
            var response = new ErrorToReturn()
            {
                StatusCode = httpContext.Response.StatusCode,
                ErrorMessage = ex.Message
            };

            await httpContext.Response.WriteAsJsonAsync(response);
        }

        private static async Task HandleErrorResponseAsync(HttpContext httpContext)
        {
            var statusCode = httpContext.Response.StatusCode;
            
            // Only process error status codes (4xx and 5xx)
            if (statusCode >= 400)
            {
                var errorMessage = statusCode switch
                {
                    StatusCodes.Status401Unauthorized => "Unauthorized. Please provide a valid token.",
                    StatusCodes.Status403Forbidden => "Forbidden. You don't have permission to access this resource.",
                    StatusCodes.Status404NotFound => $"End Point {httpContext.Request.Path} is Not Found",
                    _ => $"An error occurred with status code {statusCode}"
                };

                httpContext.Response.ContentType = "application/json";
                var response = new ErrorToReturn()
                {
                    StatusCode = statusCode,
                    ErrorMessage = errorMessage
                };

                await httpContext.Response.WriteAsJsonAsync(response);
            }
        }
    }
}
