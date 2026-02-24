

namespace ServiceLayer.Services
{
    public class CacheService : ICacheService
    {
        private readonly IRedisService _redis;

        public CacheService(IRedisService redis)
        {
            _redis = redis;
        }

        public Task<string?> GetAsync(string key)
            => _redis.GetKeyAsync(key);

        public Task RemoveAsync(string key)
            => _redis.RemoveKeyAsync(key);

        public Task SetAsync(string key, object value, TimeSpan ttl)
            => _redis.SetKeyAsync(key, value, ttl);
    }

}
