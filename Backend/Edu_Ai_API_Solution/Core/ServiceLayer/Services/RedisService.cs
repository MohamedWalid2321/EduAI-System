namespace ServiceLayer.Services
{
    
    // RedisService implements IRedisService using Upstash REST API.
    // Handles caching by storing and retrieving keys as JSON strings.
    public class RedisService : IRedisService
    {
        private readonly HttpClient _client;
        private readonly string _baseUrl;

        // Constructor initializes HttpClient and sets authorization header.

        public RedisService(IConfiguration configuration)
        {
            // Base URL for Upstash Redis REST
            _baseUrl = configuration["Redis:RestUrl"]?.TrimEnd('/')
                ?? throw new InvalidOperationException("Redis:RestUrl not configured");

            // REST token for authentication
            var token = configuration["Redis:RestToken"]
                ?? throw new InvalidOperationException("Redis:RestToken not configured");

            // Initialize HttpClient and add Authorization header
            _client = new HttpClient();
            _client.DefaultRequestHeaders.Add("Authorization", $"Bearer {token}");
        }

        /// Retrieves a cached value from Redis by key.

        public async Task<string?> GetKeyAsync(string key)
        {
            try
            {
                // URL encode key to handle special characters or slashes
                var encodedKey = Uri.EscapeDataString(key);

                var url = $"{_baseUrl}/get/{encodedKey}";

                // Send GET request
                var response = await _client.GetAsync(url);

                if (!response.IsSuccessStatusCode)
                {
                    Console.WriteLine($"Redis GET failed: {response.StatusCode} - {await response.Content.ReadAsStringAsync()}");
                    return null;
                }

                var json = await response.Content.ReadAsStringAsync();

                using var doc = JsonDocument.Parse(json);

                // Extract the "result" property
                if (doc.RootElement.TryGetProperty("result", out var result))
                {
                    return result.GetString();
                }

                // Key not found
                return null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Redis GET exception: {ex.Message}");
                return null; // fail-safe
            }
        }

        public async Task RemoveKeyAsync(string key)
        {
            try
            {
                // URL encode key to handle special characters
                var encodedKey = Uri.EscapeDataString(key);

                // Upstash REST endpoint for deleting a key
                var url = $"{_baseUrl}/del/{encodedKey}";

                // Send POST request to delete the key
                var response = await _client.PostAsync(url, null);

                // Throw exception if request failed
                response.EnsureSuccessStatusCode();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Redis DEL exception: {ex.Message}");
                throw;
            }
        }

        /// Stores a value in Redis with an optional expiration time (TTL).

        public async Task SetKeyAsync(string cacheKey, object cacheValue, TimeSpan? ttl = null)
        {
            // Serialize the object to JSON
            var json = JsonSerializer.Serialize(cacheValue);

            // URL encode key and value to avoid issues with special characters
            var encodedKey = Uri.EscapeDataString(cacheKey);
            var encodedValue = Uri.EscapeDataString(json);

            // Build URL for Upstash REST API
            string url = ttl.HasValue && ttl.Value.TotalSeconds > 0
                ? $"{_baseUrl}/set/{encodedKey}/{encodedValue}?ex={(int)ttl.Value.TotalSeconds}"
                : $"{_baseUrl}/set/{encodedKey}/{encodedValue}";

            // Send POST request
            var response = await _client.PostAsync(url, null);

            // Throw exception if request failed
            response.EnsureSuccessStatusCode();
        }
    }
}
