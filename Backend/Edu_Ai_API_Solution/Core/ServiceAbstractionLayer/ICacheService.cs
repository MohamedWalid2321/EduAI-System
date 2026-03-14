
namespace ServiceAbstractionLayer
{
    public interface ICacheService
    {
        public Task<string?> GetAsync(string key);
        public  Task SetAsync(string key, object value, TimeSpan ttl);
        public Task RemoveAsync(string key);
        public Task RemoveByPatternAsync(string pattern);
    }

    
}
