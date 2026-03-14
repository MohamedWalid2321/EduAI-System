
namespace PresentationLayer.Attributes
{
    
    // CacheAttribute automatically caches the response of an action method.
    public class CacheAttribute : ActionFilterAttribute
    {
        private readonly int _durationInSeconds;

        // Initializes a new instance of CacheAttribute.
        public CacheAttribute(int durationInSeconds = 60)
        {
            _durationInSeconds = durationInSeconds;
        }

        // Called before and after an action executes.
        // Checks cache first, executes action if no cached value, then caches the result.
        public override async Task OnActionExecutionAsync(
            ActionExecutingContext context,
            ActionExecutionDelegate next)
        {
            // Get the cache service from the DI container
            var cacheService = context.HttpContext.RequestServices.GetRequiredService<ICacheService>();

            // Generate a unique cache key based on request path and query string
            var cacheKey = GenerateCacheKey(context.HttpContext);

            // Attempt to get cached data
            try
            {
                var cachedData = await cacheService.GetAsync(cacheKey);

                if (!string.IsNullOrEmpty(cachedData))
                {
                    // Return cached response if available
                    context.Result = new ContentResult
                    {
                        Content = cachedData,
                        ContentType = "application/json",
                        StatusCode = StatusCodes.Status200OK
                    };
                    return; // Stop execution, return cached value
                }
            }
            catch (Exception ex)
            {
                // Fail-safe: log error and continue with normal execution
                Console.WriteLine($"Cache Get Error: {ex.Message}");
            }

            // Execute the action
            var executedContext = await next();

            // Cache the action result if available and no exception occurred
            if (executedContext.Exception == null &&
                executedContext.Result is ObjectResult objectResult &&
                objectResult.Value != null)
            {
                try
                {
                    await cacheService.SetAsync(
                        cacheKey,
                        objectResult.Value, // Object is serialized inside RedisService
                        TimeSpan.FromSeconds(_durationInSeconds)
                    );
                }
                catch (Exception ex)
                {
                    // Log cache set errors but don't block response
                    Console.WriteLine($"Cache Set Error: {ex.Message}");
                }
            }
        }


        /// Generates a cache key based on request path and query string.
        /// Ensures uniqueness per route and query parameters.
        private static string GenerateCacheKey(HttpContext context)
        {
            var request = context.Request;
            var keyBuilder = new StringBuilder(request.Path.ToString().ToLowerInvariant());

            // Scope cache per authenticated user so users never receive each other's data
            var userId = context.User?.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (userId != null)
                keyBuilder.Append($"|user:{userId}");

            foreach (var query in request.Query.OrderBy(q => q.Key))
            {
                keyBuilder.Append($"|{query.Key}:{query.Value}");
            }

            return keyBuilder.ToString();
        }
    }
}
