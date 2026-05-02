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
                // Use JSON-body pipeline POST to avoid %2F path-segment blocking
                // (nginx/Cloudflare reject encoded slashes in URL path segments)
                var commandJson = JsonSerializer.Serialize(new object[] { "DEL", key });
                var content = new StringContent(commandJson, Encoding.UTF8, "application/json");

                var response = await _client.PostAsync(_baseUrl, content);
                response.EnsureSuccessStatusCode();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Redis DEL exception: {ex.Message}");
                throw;
            }
        }

        // Scans for all keys matching the given glob pattern and deletes them.
        // Uses Redis SCAN to avoid blocking the server (safe for production).
        // Pattern example: "/api/course|user:*" deletes all per-user course cache entries.
        public async Task RemoveByPatternAsync(string pattern)
        {
            try
            {
                var cursor = "0";

                do
                {
                    // POST with JSON body avoids URL path-segment encoding entirely.
                    // %2F (encoded '/') in path segments is blocked by most HTTP infrastructure
                    // (nginx, Cloudflare, etc.) as path-traversal protection, breaking SCAN.
                    // JSON body carries the pattern as a plain string — no encoding issues.
                    var commandJson = JsonSerializer.Serialize(
                        new object[] { "SCAN", cursor, "MATCH", pattern, "COUNT", 100 });

                    var content = new StringContent(commandJson, Encoding.UTF8, "application/json");
                    var response = await _client.PostAsync(_baseUrl, content);
                    response.EnsureSuccessStatusCode();

                    var json = await response.Content.ReadAsStringAsync();
                    using var doc = JsonDocument.Parse(json);

                    // result[0] = next cursor (Upstash returns integer, not string)
                    // result[1] = array of matched keys
                    var result = doc.RootElement.GetProperty("result");

                    cursor = result[0].ValueKind == JsonValueKind.Number
                        ? result[0].GetInt64().ToString()
                        : result[0].GetString()!;

                    foreach (var keyElement in result[1].EnumerateArray())
                    {
                        var key = keyElement.GetString();
                        if (key != null)
                            await RemoveKeyAsync(key);
                    }
                }
                while (cursor != "0");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Redis SCAN/DEL exception: {ex.Message}");
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
