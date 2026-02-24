namespace ServiceAbstractionLayer
{
    public interface IRedisService
    {
        public Task SetKeyAsync(string cacheKey, object cacheValue, TimeSpan? ttl = null);

        public Task<string?> GetKeyAsync(string key);
        public Task RemoveKeyAsync(string key);

    }
}
